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


def is_local_ollama() -> bool:
    """Ollama's default port, per the README's own local-inference config."""
    return "11434" in settings.ai_base_url


def chat_json(model: str, text_prompt: str, image_b64: str | None = None,
              max_tokens: int = 2000, temperature: float = 0.0) -> str | None:
    """JSON-mode chat completion, optionally with one image.

    Routes through Ollama's *native* /api/chat when AI_BASE_URL is a local
    Ollama server, instead of the shared OpenAI-compatible client. Reason:
    as of Ollama 0.33.3, its OpenAI-compatible endpoint does not honor
    `think: false` for reasoning-capable models (confirmed empirically —
    qwen3-vl burns its entire token budget, sometimes 4000+, on hidden
    chain-of-thought via that endpoint and never reaches `content`, even
    with a larger num_ctx). The native API correctly suppresses thinking
    and completes in under 100 tokens — it just misfiles the final answer
    under `message.thinking` instead of `message.content` in this build,
    which this works around by checking both.

    Hosted providers (DashScope, etc.) are unaffected and keep using the
    normal OpenAI-compatible path below, unchanged.

    Returns raw text for the caller to parse (e.g. via services.llm_json),
    or None.
    """
    if not is_local_ollama():
        content: list = [{"type": "text", "text": text_prompt}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        res = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return res.choices[0].message.content

    import requests

    base = settings.ai_base_url.rsplit("/v1", 1)[0]
    message: dict = {"role": "user", "content": text_prompt}
    if image_b64:
        message["images"] = [image_b64]
    payload = {
        "model": model,
        "messages": [message],
        "think": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    r = requests.post(f"{base}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    msg = r.json().get("message", {})
    return msg.get("content") or msg.get("thinking")
