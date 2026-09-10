import base64
from unified_rag.ai_client import get_client, MODEL_VISION


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _build_image_content(image_path: str) -> dict:
    """Return an OpenAI-style vision content block for a local file path or a public URL."""
    if _is_url(image_path):
        url = image_path
    else:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        url = f"data:image/jpeg;base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}


class ImageCaptioner:
    def __init__(self):
        self.client = get_client()

    def generate_caption(self, image_path: str, metadata: dict = None) -> str:
        """
        Uses Qwen-VL to describe the image with full document context.
        Accepts both local file paths and public HTTPS URLs (e.g. Cloudinary).
        """
        metadata = metadata or {}
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown Section")
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")

        context_str = f"This image is a technical illustration labeled '{label}' on Page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It is located within the section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."

        print(f"      📸 [Vision] Sending image to {MODEL_VISION} with context: {label} (Page {page})")

        prompt = (
            f"You are a Senior Industrial Systems Engineer. {context_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Describe this specific technical component in extremely high detail.\n"
            "2. Explain its function and relationship to the surrounding assembly mentioned in the context.\n"
            "3. Identify any labels, bolts, connectors, or part numbers visible.\n"
            "4. Use professional engineering terminology. This description will be used for RAG retrieval, "
            "so include keywords that a technician would use when troubleshooting this specific part."
        )

        try:
            image_content = _build_image_content(image_path)
            response = self.client.chat.completions.create(
                model=MODEL_VISION,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            image_content,
                        ],
                    }
                ],
                max_tokens=500,
            )
            caption = response.choices[0].message.content.strip()
            return f"### {label} (Context: {section})\n\n{caption}"
        except Exception as e:
            print(f"      ❌ [Vision] API FAILED for {image_path}: {e}")
            return f"Technical diagram '{label}' on Page {page}. Description unavailable."
