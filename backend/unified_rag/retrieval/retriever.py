from sqlalchemy.orm import Session
from unified_rag.db.models import ManualChunk, InteractionMemory
from unified_rag.embeddings.embedder import embedder

class RetrievalEngine:
    def __init__(self, top_k_text=3, top_k_image=3, top_k_memory=2):
        self.top_k_text = top_k_text
        self.top_k_image = top_k_image
        self.top_k_memory = top_k_memory
        
    def retrieve(self, db: Session, query: str, manual_id: str, machine_id: str = None):
        """
        Dual-Source Vector Search:
        1. Manual Documentation (Theoretical Knowledge)
        2. Interaction Memory (Historical Field Fixes)
        """
        # 1. Embed query once
        query_emb = embedder.embed_text(query)
        
        # 2. Search Manual (Theory)
        try:
            text_results = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type.in_(["text", "table"])
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            ).limit(self.top_k_text).all()
        except:
            text_results = []
            
        # 3. Search Manual (Images)
        try:
            image_query = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type == "image"
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            )
            
            # Fetch extra to allow for deduplication
            raw_image_results = image_query.limit(self.top_k_image * 2).all()
            
            # Deduplicate by path
            image_results = []
            seen_paths = set()
            for img in raw_image_results:
                if img.path not in seen_paths:
                    seen_paths.add(img.path)
                    image_results.append(img)
                if len(image_results) >= self.top_k_image:
                    break
        except:
            image_results = []

        # 4. Search Interaction Memory (History)
        # We filter by machine_id to ensure context affinity
        historical_fixes = []
        try:
            if machine_id:
                historical_fixes = db.query(InteractionMemory).filter(
                    InteractionMemory.machine_id == machine_id
                ).order_by(
                    InteractionMemory.embedding.cosine_distance(query_emb)
                ).limit(self.top_k_memory).all()
        except Exception as e:
            print(f"Error retrieving historical fixes: {e}")
            
        return {
            "text_chunks": text_results,
            "images": image_results,
            "historical_fixes": historical_fixes
        }
