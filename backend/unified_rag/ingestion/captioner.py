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
PAGE_TEXT_PREVIEW = 900


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
        """Returns {'caption': str, 'codes': [str], 'component': str, 'kind': str}."""
        metadata = metadata or {}
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown Section")
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")
        page_text = (metadata.get("page_text") or "").strip()

        context_str = f"This is a technical illustration labelled '{label}' on page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It sits in the section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."

        print(f"      📸 [Vision] Captioning {label} (page {page}) via {MODEL_VISION}")

        b64 = _image_b64(image_path)
        if not b64:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}

        # Grounding text from the SAME page. Caught this for real: without it, a
        # diagram of a petrol two-stroke tea-pruning machine got captioned as a
        # "pneumatic reciprocating saw" — the vision model pattern-matched the
        # blade shape to a common stock category instead of using what the
        # manual itself says is on the page. A crop with no manual text nearby
        # still gets a caption; it just has less to check itself against.
        grounding = ""
        if page_text:
            snippet = " ".join(page_text.split())[:PAGE_TEXT_PREVIEW]
            grounding = (
                f"\n\nTEXT FROM THIS SAME PAGE OF THE MANUAL (ground your description in this - "
                f"do not describe a different machine, mechanism or power source than what it "
                f"says):\n\"{snippet}\"\n"
            )

        prompt = (
            f"You are a senior industrial maintenance engineer. {context_str}{grounding}\n\n"
            "Study the image and return a JSON object with exactly these keys:\n"
            '  "kind": what this image IS - one of "diagram", "schematic", "chart", '
            '"flowchart", "exploded_view", "photo", "table".\n'
            '  "component": short name of the main component shown (3-6 words).\n'
            '  "codes": array of EVERY numbered or coded callout printed on the image, including '
            'plain numbers in circles or with leader lines (e.g. "1", "2", ... up to whatever the '
            'highest number visible is), plus any alphanumeric codes (e.g. "P-100", "V-220"). Scan '
            'the whole image systematically left-to-right, top-to-bottom - do not stop after the '
            'first few. Codes only, no descriptions.\n'
            '  "leader_labels": array of any TEXT labels connected to the diagram by a leader line '
            'or pointer (e.g. "Harness attachment point", "Drive shaft tube") - the exact wording '
            'shown, not numbered callouts (those go in "codes").\n'
            '  "caption": a dense 100-180 word description written for search retrieval. Cover: '
            'what the component is, its function, how its parts connect, and what EACH numbered '
            'callout or leader label points to if you can tell from the drawing. Use the page text '
            'above to get the machine, power source and terminology right. Use standard engineering '
            'terminology a technician would search for.\n\n'
            "Describe only what is actually visible. Do not invent part numbers, and do not guess "
            "a different type of machine or power source than the page text states."
        )

        raw = None
        try:
            # 1200 was tight once "leader_labels" and "scan the whole image
            # systematically" were added - a dense composite with a dozen
            # callouts needs room to actually enumerate them all rather than
            # stopping at the first few.
            raw = chat_json(MODEL_VISION, prompt, image_b64=b64, max_tokens=1800, temperature=0.1)
            data = loads_tolerant(raw) if raw else None
        except Exception as e:
            print(f"      ⚠️ [Vision] Structured caption failed for {label}: {e}")
            data = None

        VALID_KINDS = {"diagram", "schematic", "chart", "flowchart",
                       "exploded_view", "photo", "table"}
        if isinstance(data, dict) and data.get("caption"):
            component = str(data.get("component") or label).strip()
            codes = [str(c).strip() for c in (data.get("codes") or []) if str(c).strip()]
            leader_labels = [str(c).strip() for c in (data.get("leader_labels") or []) if str(c).strip()]
            caption = str(data["caption"]).strip()
            kind = str(data.get("kind", "")).strip().lower()
            kind = kind if kind in VALID_KINDS else "diagram"
        else:
            # Structured mode failed — fall back to a plain prose caption rather
            # than storing an empty chunk that can never be retrieved.
            print(f"      ↩️ [Vision] Falling back to prose caption for {label}")
            caption = (chat_text(
                MODEL_VISION,
                f"{context_str}{grounding}\n\nDescribe this technical component in 80-150 words for a "
                "maintenance technician: what it is, its function, visible labels and part codes, "
                "and the faults this diagram helps diagnose.",
                image_b64=b64, max_tokens=900, temperature=0.2,
            ) or "").strip()
            component, codes, leader_labels, kind = label, [], [], "diagram"

        if not caption:
            return {"caption": "", "codes": [], "component": label, "kind": "diagram"}

        pretty_kind = kind.replace("_", " ").title()
        header = f"### {component} ({pretty_kind}) — page {page}"
        if section != "Unknown Section":
            header += f" (section: {section})"
        code_line = f"\nCallout codes shown: {', '.join(codes)}" if codes else ""
        label_line = f"\nLeader-line labels: {', '.join(leader_labels)}" if leader_labels else ""
        return {
            "caption": f"{header}\n\n{caption}{code_line}{label_line}",
            # Both feed services/callout_linker.py: numeric/alphanumeric codes
            # resolve against "(1)"/"item 1"-style manual text, free-text leader
            # labels resolve as literal phrase matches against the same text.
            "codes": codes + leader_labels,
            "component": component,
            "kind": kind,
        }

    def generate_caption(self, image_path: str, metadata: dict = None) -> str:
        """Back-compatible string form."""
        return self.describe(image_path, metadata)["caption"]
