from unified_rag.ai_client import get_client, MODEL_EMBEDDING, EMBEDDING_DIM

class MultimodalEmbedder:
    """Unified embedding space for text, tables, and vision-generated image captions."""

    def embed_text(self, text: str) -> list[float]:
        response = get_client().embeddings.create(
            input=text,
            model=MODEL_EMBEDDING,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding

embedder = MultimodalEmbedder()
