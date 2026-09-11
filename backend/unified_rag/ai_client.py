"""
Centralized AI client for chat, vision, and embedding calls.

Every LLM/embedding call in this codebase goes through get_client() and the
MODEL_* constants below instead of constructing its own OpenAI client or
hardcoding a model string. This is what makes swapping providers (DashScope
now, self-hosted vLLM/Ollama later) a single .env change instead of an
N-file migration.
"""
import re
from functools import lru_cache
from typing import Optional

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


def _ollama_native(model: str, prompt: str, image_b64: Optional[str], max_tokens: int,
                   temperature: float, json_mode: bool, system: Optional[str]) -> Optional[str]:
    """Call Ollama's native /api/chat.

    Why this exists: as of Ollama 0.33.3 the OpenAI-compatible endpoint does
    NOT honor `think: false` for reasoning models. Verified empirically —
    qwen3-vl burns its entire budget (4000+ tokens, even with num_ctx raised)
    on hidden chain-of-thought and never emits `content`. The native endpoint
    suppresses thinking correctly and answers in under 100 tokens; it just
    misfiles the answer under `message.thinking` in this build, so we read
    both fields.
    """
    import requests

    base = settings.ai_base_url.rsplit("/v1", 1)[0]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_msg: dict = {"role": "user", "content": prompt}
    if image_b64:
        user_msg["images"] = [image_b64]
    messages.append(user_msg)

    payload = {
        "model": model,
        "messages": messages,
        "think": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"

    r = requests.post(f"{base}/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    msg = r.json().get("message", {})
    return _strip_reasoning(msg.get("content")) or _strip_reasoning(msg.get("thinking"))


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: Optional[str]) -> Optional[str]:
    """Remove a Qwen reasoning block from a response.

    `think: False` suppresses this almost always, but not reliably: three
    captions in a real ingested manual were stored with the model's raw
    reasoning as their body ("<think> Got it, let's tackle this. First, the
    user wants..."), which then got embedded and would have been shown to a
    technician as the figure's description.

    An UNCLOSED <think> means the model spent its whole budget reasoning and
    never reached an answer — everything after the tag is reasoning, so the
    result is nothing, and the caller's fallback path (a plain prose caption,
    or skipping the chunk) is the correct outcome rather than storing this.
    """
    if not text:
        return None
    cleaned = _THINK_UNCLOSED.sub("", _THINK_BLOCK.sub("", text)).strip()
    return cleaned or None


def _openai_compat(model: str, prompt: str, image_b64: Optional[str], max_tokens: int,
                   temperature: float, json_mode: bool, system: Optional[str]) -> Optional[str]:
    content: list = [{"type": "text", "text": prompt}]
    if image_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    res = get_client().chat.completions.create(**kwargs)
    return _strip_reasoning(res.choices[0].message.content)


def chat_text(model: str, prompt: str, image_b64: Optional[str] = None, max_tokens: int = 800,
              temperature: float = 0.2, system: Optional[str] = None) -> Optional[str]:
    """Plain-text completion, optionally with one image.

    Routes to Ollama's native API when running locally (see _ollama_native for
    why); hosted providers keep the normal OpenAI-compatible path unchanged.
    """
    fn = _ollama_native if is_local_ollama() else _openai_compat
    return fn(model, prompt, image_b64, max_tokens, temperature, False, system)


def chat_json(model: str, text_prompt: str, image_b64: Optional[str] = None, max_tokens: int = 2000,
              temperature: float = 0.0, system: Optional[str] = None) -> Optional[str]:
    """JSON-mode completion, optionally with one image. Returns raw text for the
    caller to parse (e.g. via services.llm_json.loads_tolerant), or None."""
    fn = _ollama_native if is_local_ollama() else _openai_compat
    return fn(model, text_prompt, image_b64, max_tokens, temperature, True, system)
