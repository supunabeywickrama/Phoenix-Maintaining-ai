from unified_rag.ingestion.parser import DocumentParser
from unified_rag.ingestion.chunker import ContextualChunker
from unified_rag.embeddings.embedder import embedder
from unified_rag.ingestion.captioner import ImageCaptioner
from services.table_transformer import TableTransformer
from services.callout_linker import link_image_chunks
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk
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

    # 1. Structural Parsing (includes YOLOv8 + validation + figure splitting)
    print(f"🔍 [Pipeline] Stage 1/5: Multi-modal Structural Parsing...")
    parsed_data = parser.parse_pdf(file_path, manual_id)

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
            meta["codes"] = result["codes"]
            meta["component"] = result["component"]
            meta["kind"] = result.get("kind", "diagram")
            chunk["kind"] = result.get("kind", "diagram")
            chunk["metadata"] = meta
        elif chunk["type"] == "table":
            ctx = chunk.get("metadata", {}).get("section", "Technical Data")
            chunk["content"] = await asyncio.to_thread(
                tabler.summarize_table, chunk["content"], ctx
            )
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

    # 5. Embedding & Storage
    print(f"💾 [Pipeline] Stage 5/5: Embedding and persistence...")

    stored = 0
    batch_size = 20
    for i in range(0, len(enriched_chunks), batch_size):
        batch = enriched_chunks[i:i + batch_size]
        print(f"   ∟ Committing batch {i//batch_size + 1}...")

        db = SessionLocal()
        try:
            for chunk in batch:
                if not chunk["content"]: continue

                emb = embedder.embed_text(chunk["content"])
                db_chunk = ManualChunk(
                    manual_id=chunk["manual_id"],
                    type=chunk["type"],
                    content=chunk["content"],
                    path=chunk.get("path"),
                    embedding=emb,
                    page=chunk["page"],
                    kind=chunk.get("kind"),
                    render_markdown=chunk.get("render_markdown"),
                    figure_role=chunk.get("figure_role"),
                    parent_path=chunk.get("parent_path"),
                    width=chunk.get("width"),
                    height=chunk.get("height"),
                )
                db.add(db_chunk)
                stored += 1
            db.commit()
        except Exception as e:
            print(f"   ⚠️ [Pipeline] Batch commit failed: {e}")
            db.rollback()
        finally:
            db.close()

    print(f"✨ [Pipeline v2] SUCCESS: {manual_id} fully ingested ({stored} chunks stored).")
    return {"status": "success", "chunks": stored}

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
