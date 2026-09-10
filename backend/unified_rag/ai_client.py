"""
Centralized AI client for chat, vision, and embedding calls.

Every LLM/embedding call in this codebase goes through get_client() and the
MODEL_* constants below instead of constructing its own OpenAI client or
hardcoding a model string. This is what makes swapping providers (DashScope
now, self-hosted vLLM/Ollama later) a single .env change instead of an
N-file migration.
"""
from functools import lru_cache
from openai import OpenAI
from unified_rag.config import settings

MODEL_CHAT = settings.model_chat
MODEL_CHAT_LIGHT = settings.model_chat_light
MODEL_VISION = settings.model_vision
MODEL_EMBEDDING = settings.model_embedding
EMBEDDING_DIM = settings.embedding_dim


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=settings.ai_api_key, base_url=settings.ai_base_url)
