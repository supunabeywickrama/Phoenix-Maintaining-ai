"""
Vision captioning for extracted figures.

Produces a retrieval-optimised description AND the raw callout codes printed on
the diagram (e.g. "P-100", "7", "V-220"). Industrial drawings routinely label
parts with a bare code and a leader line, with the actual part name living in a
table or paragraph elsewhere in the manual — services/callout_linker.py resolves
those codes afterwards so a technician searching the part *name* still finds the
diagram.
"""
import base64
import json
import os
from pathlib import Path
from typing import Optional

from unified_rag.ai_client import chat_json, chat_text, MODEL_VISION
from services.llm_json import loads_tolerant

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _resolve_local(path: str) -> Path:
    """Local figures are stored as a repo-relative 'data/...' path (see
    CloudinaryService._upload_local). Resolve against the backend dir so this
    works regardless of the process's working directory."""
    p = Path(path)
    return p if p.is_absolute() else (BACKEND_DIR / p)


def _image_b64(image_path: str) -> Optional[str]:
    """Base64 for the vision call. Fetches URLs, reads local paths."""
    try:
        if _is_url(image_path):
            import requests
            r = requests.get(image_path, timeout=30)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        with open(_resolve_local(image_path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"      ❌ [Vision] Could not load image {image_path}: {e}")
        return None


class ImageCaptioner:
    def describe(self, image_path: str, metadata: dict = None) -> dict:
        """Returns {'caption': str, 'codes': [str], 'component': str}."""
        metadata = metadata or {}
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown Section")
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")

        context_str = f"This is a technical illustration labelled '{label}' on page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It sits in the section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."

        print(f"      📸 [Vision] Captioning {label} (page {page}) via {MODEL_VISION}")

        b64 = _image_b64(image_path)
        if not b64:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}

        prompt = (
            f"You are a senior industrial maintenance engineer. {context_str}\n\n"
            "Study the image and return a JSON object with exactly these keys:\n"
            '  "kind": what this image IS - one of "diagram", "schematic", "chart", '
            '"flowchart", "exploded_view", "photo", "table".\n'
            '  "component": short name of the main component shown (3-6 words).\n'
            '  "codes": array of every part/callout code printed on the image '
            '(e.g. "P-100", "V-220", "7", "3a"). Empty array if none. Codes only — no descriptions.\n'
            '  "caption": a dense 80-150 word description written for search retrieval. Cover: '
            'what the component is, its function, how its parts connect, any visible labels, ports, '
            'fasteners or wear points, and the symptoms/faults a technician would be looking at this '
            'diagram to diagnose. Use standard engineering terminology a technician would search for.\n\n'
            "Describe only what is actually visible. Do not invent part numbers."
        )

        raw = None
        try:
            raw = chat_json(MODEL_VISION, prompt, image_b64=b64, max_tokens=1200, temperature=0.1)
            data = loads_tolerant(raw) if raw else None
        except Exception as e:
            print(f"      ⚠️ [Vision] Structured caption failed for {label}: {e}")
            data = None

        VALID_KINDS = {"diagram", "schematic", "chart", "flowchart",
                       "exploded_view", "photo", "table"}
        if isinstance(data, dict) and data.get("caption"):
            component = str(data.get("component") or label).strip()
            codes = [str(c).strip() for c in (data.get("codes") or []) if str(c).strip()]
            caption = str(data["caption"]).strip()
            kind = str(data.get("kind", "")).strip().lower()
            kind = kind if kind in VALID_KINDS else "diagram"
        else:
            # Structured mode failed — fall back to a plain prose caption rather
            # than storing an empty chunk that can never be retrieved.
            print(f"      ↩️ [Vision] Falling back to prose caption for {label}")
            caption = (chat_text(
                MODEL_VISION,
                f"{context_str}\n\nDescribe this technical component in 80-150 words for a "
                "maintenance technician: what it is, its function, visible labels and part codes, "
                "and the faults this diagram helps diagnose.",
                image_b64=b64, max_tokens=900, temperature=0.2,
            ) or "").strip()
            component, codes, kind = label, [], "diagram"

        if not caption:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}

        pretty_kind = kind.replace("_", " ").title()
        header = f"### {component} ({pretty_kind}) — page {page}"
        if section != "Unknown Section":
            header += f" (section: {section})"
        code_line = f"\nCallout codes shown: {', '.join(codes)}" if codes else ""
        return {
            "caption": f"{header}\n\n{caption}{code_line}",
            "codes": codes,
            "component": component,
            "kind": kind,
        }

    def generate_caption(self, image_path: str, metadata: dict = None) -> str:
        """Back-compatible string form."""
        return self.describe(image_path, metadata)["caption"]
