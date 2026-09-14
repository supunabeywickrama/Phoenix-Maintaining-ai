"""
Figure validation — vision decides, never pixel counts.

The layout model tags a region 'Picture', but that bucket also catches logos,
warning icons, page furniture and text blocks. Whether a region is a figure a
technician would use is judged by the vision model looking at it.

Pixel rules (minimum size, ink ratio, aspect ratio, "looks like text") used to
drop regions here. They threw away real content: on the Cab Windows exploded
view (873 loader parts manual, p.226) they dropped over 40 pieces of the drawing
as "too small — icon or glyph", because a parts drawing is made of many small
line-art parts. Size says nothing about whether a part matters.

The same vision call also says how the figure is laid out, which decides
whether it may be split at all:
  - exploded_parts:  one drawing with many numbered parts (a parts-catalogue
                     page). Always kept whole — cutting it apart separates the
                     callout numbers from the parts they point at.
  - composite_views: separate sub-drawings side by side, e.g. "(a) side view"
                     and "(b) top view". These may be split into their views.
  - single:          one drawing. Kept whole.
"""
import base64

import cv2
import numpy as np

from unified_rag.ai_client import chat_json, MODEL_VISION
from services.llm_json import loads_tolerant

# Labels that mean "not a figure a technician would use", regardless of what
# the model puts in is_figure.
REJECT_KINDS = {"text", "icon", "logo", "decoration", "blank"}
LAYOUTS = {"single", "exploded_parts", "composite_views"}

# Longest side sent to the vision model. Only for the judgement call — the
# stored figure keeps its full resolution.
CLASSIFY_MAX_SIDE = 1400


def _encode(crop: np.ndarray) -> str:
    h, w = crop.shape[:2]
    scale = min(1.0, CLASSIFY_MAX_SIDE / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def classify_figure(crop: np.ndarray, page_text: str = "") -> dict:
    """{'keep': bool, 'kind': str, 'layout': str, 'reason': str}.

    Fails open: if the vision call breaks, the figure is kept whole. Losing a
    real diagram because the validator crashed is the worst outcome here.
    """
    if crop is None or crop.size == 0:
        # Nothing was cut out — there is no image to judge or store.
        return {"keep": False, "kind": "blank", "layout": "single", "reason": "empty crop"}

    b64 = _encode(crop)
    if not b64:
        return {"keep": True, "kind": "diagram", "layout": "single", "reason": "encode failed — keeping"}

    context = ""
    if page_text:
        context = f'\nText printed on the same page (for context): "{" ".join(page_text.split())[:500]}"\n'
    prompt = (
        "Classify this region cut from a machine manual page." + context + "\n"
        "Return JSON with keys:\n"
        '  "is_figure": true if it is a technical illustration a technician would use: a '
        "diagram, exploded parts view, schematic, wiring/hydraulic/flow diagram, chart, or "
        "photo of equipment. Small or thin line drawings still count.\n"
        '  "kind": one of "diagram", "exploded_view", "schematic", "chart", "flowchart", '
        '"photo", "text", "icon", "logo", "decoration", "blank".\n'
        '  "layout": one of\n'
        '     "exploded_parts"  - ONE drawing of an assembly with many parts, usually with '
        "numbered callouts (a parts-catalogue page), even if the parts are spread apart;\n"
        '     "composite_views" - two or more SEPARATE drawings of different views, e.g. '
        "labelled (a) and (b);\n"
        '     "single"          - one drawing of one thing.\n'
        '  "reason": under 12 words.\n'
        "is_figure is false only for plain text or paragraphs, company logos, a lone warning "
        "icon, page borders or blank areas. A rectangular frame drawn around a figure is part "
        "of the figure."
    )
    try:
        data = loads_tolerant(chat_json(MODEL_VISION, prompt, image_b64=b64,
                                        max_tokens=250, temperature=0.0) or "")
    except Exception as e:
        print(f"      ⚠️ [Validator] Vision check failed: {e}")
        data = None
    if not isinstance(data, dict) or "is_figure" not in data:
        return {"keep": True, "kind": "diagram", "layout": "single",
                "reason": "vision check unavailable — keeping whole"}

    kind = str(data.get("kind", "")).strip().lower() or "diagram"
    layout = str(data.get("layout", "")).strip().lower()
    layout = layout if layout in LAYOUTS else "single"
    keep = bool(data["is_figure"])
    # The model sometimes answers is_figure=true while labelling the region
    # "logo" or "icon"; the specific label is the more reliable signal.
    if kind in REJECT_KINDS:
        keep = False
    return {"keep": keep, "kind": kind, "layout": layout,
            "reason": str(data.get("reason", "")).strip()[:120]}


def is_valid_figure(crop: np.ndarray, page_text: str = "") -> tuple:
    """(keep, reason) — vision only. Kept for callers that only need yes/no."""
    verdict = classify_figure(crop, page_text)
    return verdict["keep"], f"{verdict['kind']}/{verdict['layout']}: {verdict['reason']}"
