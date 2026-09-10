from unified_rag.ingestion.parser import DocumentParser
from unified_rag.ingestion.chunker import ContextualChunker
from unified_rag.embeddings.embedder import embedder
from unified_rag.ingestion.captioner import ImageCaptioner
from services.table_transformer import TableTransformer
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk
import asyncio
import time

async def process_manual_async(file_path: str, manual_id: str):
    """
    Ingestion Pipeline v2: Structural Parsing -> Concurrent Enrichment -> Parallel Embedding.
    """
    print(f"🏗️ [Pipeline v2] Starting high-precision ingestion for {manual_id}...")
    parser = DocumentParser()
    chunker = ContextualChunker()
    captioner = ImageCaptioner()
    tabler = TableTransformer()
    
    # 1. Structural Parsing (includes YOLOv8 + Figure Splitting)
    print(f"🔍 [Pipeline] Stage 1/4: Multi-modal Structural Parsing...")
    parsed_data = parser.parse_pdf(file_path, manual_id)
    
    # 2. Adaptive Chunking
    print(f"✂️ [Pipeline] Stage 2/4: Semantic recursive chunking...")
    chunks = chunker.chunk_data(parsed_data, manual_id)
    print(f"✅ [Pipeline] Created {len(chunks)} structural chunks.")
    
    # 3. Concurrent Enrichment (LLM-based)
    print(f"🧠 [Pipeline] Stage 3/4: Concurrent LLM Enrichment (Captions + Tables)...")
    
    async def enrich_chunk(chunk):
        if chunk["type"] == "image":
            # Pass full metadata for context-aware prompting
            chunk["content"] = await asyncio.to_thread(
                captioner.generate_caption, chunk["path"], chunk.get("metadata")
            )
        elif chunk["type"] == "table":
            # Pass section context for table summarization
            ctx = chunk.get("metadata", {}).get("section", "Technical Data")
            chunk["content"] = await asyncio.to_thread(
                tabler.summarize_table, chunk["content"], ctx
            )
        return chunk

    # Process enrichments in managed batches to respect rate limits
    enriched_chunks = []
    sem = asyncio.Semaphore(10) # Limit concurrent API calls
    
    async def safe_enrich(c):
        async with sem:
            return await enrich_chunk(c)

    enriched_chunks = await asyncio.gather(*[safe_enrich(c) for c in chunks])
    
    # 4. Embedding & Storage
    print(f"💾 [Pipeline] Stage 4/4: Embedding and persistence...")
    
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
                    page=chunk["page"]
                )
                db.add(db_chunk)
            db.commit()
        except Exception as e:
            print(f"   ⚠️ [Pipeline] Batch commit failed: {e}")
            db.rollback()
        finally:
            db.close()

    print(f"✨ [Pipeline v2] SUCCESS: {manual_id} fully ingested.")
    return {"status": "success", "chunks": len(enriched_chunks)}

def process_manual(file_path: str, manual_id: str):
    """Sync wrapper for the async pipeline. Handles existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # If in a running loop (like FastAPI), we can't use asyncio.run()
        # The caller (api/endpoints.py) should ideally await process_manual_async directly.
        # But for safety, we provide this warning.
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(process_manual_async(file_path, manual_id))
    else:
        return asyncio.run(process_manual_async(file_path, manual_id))
