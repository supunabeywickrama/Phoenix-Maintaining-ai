"""
smoke_test_qwen.py — Tier 0 verification for Phoenix: standalone checks against DashScope
for each of the 4 model roles, no Postgres/FastAPI required.

Run this first after setting AI_API_KEY in .env, before testing any app code.
Catches base_url/auth/model-name problems early and confirms:
  - qwen-max: chat completion + JSON mode (DashScope requires the literal
    word "json" somewhere in the prompt for response_format=json_object)
  - qwen-flash: plain-text chat completion
  - qwen-vl-max: vision call against a real image
  - text-embedding-v4: embedding call returns a vector of the configured length

Usage (from backend/):
    python scripts/smoke_test_qwen.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from unified_rag.ai_client import get_client, MODEL_CHAT, MODEL_CHAT_LIGHT, MODEL_VISION, MODEL_EMBEDDING, EMBEDDING_DIM

# The vision check only proves the call path and auth work, not caption quality,
# so it generates its own image rather than depending on a file in the repo.
SAMPLE_IMAGE = None


def _sample_image_bytes() -> bytes:
    """A small PNG with a couple of shapes — enough for the model to describe."""
    import numpy as np
    import cv2

    img = np.full((240, 320, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (40, 60), (150, 180), (60, 60, 200), -1)
    cv2.circle(img, (230, 120), 55, (60, 160, 60), -1)
    cv2.putText(img, "PHOENIX", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Could not encode the sample image")
    return buf.tobytes()


def check_chat_json():
    print(f"[1/4] {MODEL_CHAT} chat + JSON mode ...", end=" ", flush=True)
    client = get_client()
    res = client.chat.completions.create(
        model=MODEL_CHAT,
        messages=[{"role": "user", "content": "Reply with a JSON object: {\"ok\": true}"}],
        response_format={"type": "json_object"},
        max_tokens=50,
    )
    print("OK ->", res.choices[0].message.content.strip())


def check_chat_light():
    print(f"[2/4] {MODEL_CHAT_LIGHT} plain-text chat ...", end=" ", flush=True)
    client = get_client()
    res = client.chat.completions.create(
        model=MODEL_CHAT_LIGHT,
        messages=[{"role": "user", "content": "Say hello in exactly two words."}],
        max_tokens=20,
    )
    print("OK ->", res.choices[0].message.content.strip())


def check_vision():
    print(f"[3/4] {MODEL_VISION} vision ...", end=" ", flush=True)
    import base64
    b64 = base64.b64encode(_sample_image_bytes()).decode("utf-8")
    client = get_client()
    res = client.chat.completions.create(
        model=MODEL_VISION,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        max_tokens=100,
    )
    print("OK ->", res.choices[0].message.content.strip())


def check_embedding():
    print(f"[4/4] {MODEL_EMBEDDING} embedding (dim={EMBEDDING_DIM}) ...", end=" ", flush=True)
    client = get_client()
    res = client.embeddings.create(
        model=MODEL_EMBEDDING,
        input="Pump bearing temperature exceeded 92C with sustained motor current rise.",
        dimensions=EMBEDDING_DIM,
    )
    vec = res.data[0].embedding
    assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}-dim vector, got {len(vec)}"
    print(f"OK -> vector length {len(vec)}")


if __name__ == "__main__":
    for check in (check_chat_json, check_chat_light, check_vision, check_embedding):
        try:
            check()
        except Exception as e:
            print(f"FAILED -> {e}")
