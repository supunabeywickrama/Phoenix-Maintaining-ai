from unified_rag.ingestion.parser import DocumentParser
from unified_rag.ingestion.chunker import ContextualChunker
from unified_rag.embeddings.embedder import embedder
from unified_rag.ingestion.captioner import ImageCaptioner
from services.table_transformer import TableTransformer
from services.callout_linker import link_image_chunks
from unified_rag.db import qdrant_store as qs
import asyncio
import time

async def process_manual_async(file_path: str, manual_id: str):
    """
    Ingestion Pipeline v2: Structural Parsing -> Concurrent Enrichment ->
    Callout Linking -> Parallel Embedding.
    """
    print(f"🏗️ [Pipeline v2] Starting high-precision ingestion for {manual_id}...")
    parser = DocumentParser()
    chunker = ContextualChunker()
    captioner = ImageCaptioner()
    tabler = TableTransformer()

    # 0. Page preparation: scanned / photographed pages are made upright,
    #    cropped to the sheet and straightened. Digital pages are untouched and
    #    page numbers stay 1:1 with the uploaded PDF.
    from services.page_prep import prepare_pdf
    print(f"🧭 [Pipeline] Stage 0/5: Preparing scanned pages...")
    prepared_path, _fixes = prepare_pdf(file_path)

    # 1. Structural Parsing (includes YOLOv8 + validation + figure splitting)
    print(f"🔍 [Pipeline] Stage 1/5: Multi-modal Structural Parsing...")
    try:
        parsed_data = parser.parse_pdf(prepared_path, manual_id)
    finally:
        if prepared_path != file_path:
            import os
            try:
                os.remove(prepared_path)
            except OSError:
                pass

    # 2. Adaptive Chunking
    print(f"✂️ [Pipeline] Stage 2/5: Semantic recursive chunking...")
    chunks = chunker.chunk_data(parsed_data, manual_id)
    print(f"✅ [Pipeline] Created {len(chunks)} structural chunks.")

    # Snapshot the raw text/table content BEFORE enrichment. Table summarisation
    # is prose and can drop the very part codes callout linking searches for, so
    # the linker gets the original text.
    raw_corpus = [
        {"type": c["type"], "content": c.get("content"), "page": c.get("page")}
        for c in chunks if c["type"] in ("text", "table") and c.get("content")
    ]

    # 3. Concurrent Enrichment (LLM-based)
    print(f"🧠 [Pipeline] Stage 3/5: Concurrent LLM Enrichment (Captions + Tables)...")

    async def enrich_chunk(chunk):
        if chunk["type"] == "image":
            # The page lives on the chunk, but the captioner prompts with it —
            # without this the caption literally reads "page Unknown".
            meta = dict(chunk.get("metadata") or {})
            meta["page"] = chunk.get("page")
            result = await asyncio.to_thread(captioner.describe, chunk["path"], meta)
            chunk["content"] = result["caption"]
            meta["parts_resolved"] = result.get("parts_resolved", False)
            meta["codes"] = result["codes"]
            meta["component"] = result["component"]
            meta["kind"] = result.get("kind", "diagram")
            chunk["kind"] = result.get("kind", "diagram")
            chunk["metadata"] = meta
        elif chunk["type"] == "table" and (chunk.get("metadata") or {}).get("parts_list"):
            # A parts list is already clean, readable text straight from the
            # printed page. An LLM summary adds nothing and can only paraphrase
            # a part number wrong, so it is stored as read.
            meta = chunk["metadata"]
            chunk["content"] = f"### Table — {meta.get('title', 'Parts list')}\n\n{chunk['content']}"
        elif chunk["type"] == "table":
            meta = chunk.get("metadata") or {}
            # parser.py derives this from the table's own caption/header rather
            # than the section, which leaks across everything on the page.
            ctx = meta.get("title") or meta.get("section") or "Technical Data"
            rows = chunk["content"]  # readable "Item: 3 | Component: ..." lines
            summary = await asyncio.to_thread(tabler.summarize_table, rows, ctx)
            # The header is written here, not by the LLM: rag.py's _title_of()
            # takes the first line of the content as the display title, and an
            # LLM asked to open its own summary tends to invent a plausible
            # manual-style heading ("Figure 1.7: ...") for a table that has no
            # real one. A deterministic header can't hallucinate.
            # Page is not repeated here: it is already carried on the chunk's own
            # `page` column and shown separately everywhere this title is used.
            #
            # The ROWS ARE KEPT under the summary, not replaced by it. A prose
            # summary of a legend destroys exactly the thing a legend is for:
            # the real output for "Key to Figure 1.1" read "Columns 2 through 7
            # contain associated numerical identifiers (ranging from 8 to 12)"
            # followed by an unordered list of component names - so nothing in
            # the embedded text said item 3 was the clutch housing, and no
            # search for a part name could land on the figure it belongs to.
            # The summary still earns its place for conceptual matches; the
            # rows are what make exact lookups work.
            chunk["content"] = f"### Table — {ctx}\n\n{summary}\n\n{rows}"
        return chunk

    sem = asyncio.Semaphore(10)  # Limit concurrent API calls

    async def safe_enrich(c):
        async with sem:
            return await enrich_chunk(c)

    enriched_chunks = await asyncio.gather(*[safe_enrich(c) for c in chunks])

    # 4. Callout Linking — resolve codes printed on diagrams ("P-100", "7")
    #    against the manual's own text/tables so figures are findable by part name.
    print(f"🔗 [Pipeline] Stage 4/5: Linking diagram callouts to part names...")
    try:
        linkable = list(enriched_chunks) + raw_corpus
        enriched_count = link_image_chunks(linkable)
        print(f"✅ [Pipeline] Enriched {enriched_count} figures with resolved part names.")
    except Exception as e:
        # Linking is an enhancement; never let it sink an otherwise good ingest.
        print(f"⚠️ [Pipeline] Callout linking skipped: {e}")

    # 5. Embedding & Storage (Qdrant — see unified_rag/db/qdrant_store.py)
    print(f"💾 [Pipeline] Stage 5/5: Embedding and persistence...")

    stored = 0
    empty = 0
    failures = []  # (page, type, reason) — surfaced to the caller, not just printed
    buffer = []
    BATCH = qs.UPSERT_BATCH

    def flush():
        """One request per batch instead of one per chunk. On failure the batch
        is retried point-by-point so a single bad chunk cannot take the other
        63 down with it."""
        nonlocal stored, buffer
        if not buffer:
            return
        try:
            stored += qs.upsert_manual_chunks(buffer)
        except Exception as e:
            print(f"   ⚠️ [Pipeline] Batch upsert failed ({e}); retrying individually...")
            for item in buffer:
                try:
                    stored += qs.upsert_manual_chunks([item])
                except Exception as inner:
                    failures.append((item.get("page"), item.get("type"), str(inner)))
                    print(f"   ⚠️ [Pipeline] Failed to store chunk (page {item.get('page')}): {inner}")
        buffer = []

    for i, chunk in enumerate(enriched_chunks):
        if not chunk["content"]:
            empty += 1
            continue
        try:
            emb = embedder.embed_text(chunk["content"])
        except Exception as e:
            failures.append((chunk.get("page"), chunk.get("type"), f"embedding failed: {e}"))
            print(f"   ⚠️ [Pipeline] Could not embed chunk {i} (page {chunk.get('page')}): {e}")
            continue

        buffer.append({
            "vector": emb,
            "manual_id": chunk["manual_id"],
            "type": chunk["type"],
            "content": chunk["content"],
            "path": chunk.get("path"),
            "page": chunk["page"],
            "kind": chunk.get("kind"),
            "render_markdown": chunk.get("render_markdown"),
            "figure_role": chunk.get("figure_role"),
            "parent_path": chunk.get("parent_path"),
            "width": chunk.get("width"),
            "height": chunk.get("height"),
            # Parts-list entries: exact-match lookup by part number, which a
            # vector alone does badly on codes like "17C-612".
            "ref": (chunk.get("metadata") or {}).get("ref"),
            # A figure's letter-number item codes ("B08-13") share the exact-lookup
            # field, so a search for a code printed on a drawing finds the drawing.
            "part_numbers": (chunk.get("metadata") or {}).get("part_numbers")
                            or (chunk.get("metadata") or {}).get("item_codes"),
        })
        if len(buffer) >= BATCH:
            flush()
            print(f"   ∟ Stored {stored}/{len(enriched_chunks)}...")
    flush()

    # A partially-broken ingest used to look identical to a clean one from the
    # frontend: every failure was a print() in a server log nobody reads.
    if failures:
        print(f"⚠️ [Pipeline] {len(failures)} chunk(s) did not make it into the index:")
        for page, ctype, reason in failures[:10]:
            print(f"     - page {page} ({ctype}): {reason[:160]}")

    print(f"✨ [Pipeline v2] {manual_id} ingested: {stored} stored, "
          f"{len(failures)} failed, {empty} empty.")
    return {
        "status": "success" if not failures else "partial",
        "chunks": stored,
        "failed": len(failures),
        "empty": empty,
        "failure_samples": [
            {"page": p, "type": t, "reason": r[:200]} for p, t, r in failures[:5]
        ],
    }

def process_manual(file_path: str, manual_id: str):
    """Sync wrapper for the async pipeline. Handles existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(process_manual_async(file_path, manual_id))
    else:
        return asyncio.run(process_manual_async(file_path, manual_id))
