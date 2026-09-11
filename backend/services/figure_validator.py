"""
Figure validation — reject what isn't actually a diagram.

The layout model tags a region 'Picture', but in real manuals that bucket also
catches headers/footers rendered as graphics, logos, warning icons, decorative
rules, and dense text blocks. Captioning those wastes a vision call per figure
and, worse, pollutes retrieval with image chunks that show a technician nothing.

Two tiers, cheapest first:
  1. Pixel heuristics (free): size, aspect ratio, ink density, text-likeness.
  2. A vision check (one call) only for crops the heuristics can't settle.
"""
import cv2
import numpy as np

from unified_rag.ai_client import chat_json, MODEL_VISION
from services.llm_json import loads_tolerant

# An icon or a stray rule is small in absolute terms; real diagrams aren't.
MIN_AREA_PX = 14_000          # ~120x120
MIN_SIDE_PX = 60
MAX_ASPECT = 12.0             # separator rules / text strips
MIN_INK_RATIO = 0.008         # effectively blank
MAX_INK_RATIO = 0.97          # a solid filled block, e.g. a colour bar

# Labels that mean "not a figure a technician would use", regardless of what the
# model puts in is_diagram.
REJECT_KINDS = {"text", "icon", "logo", "decoration", "blank"}


def _ink_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return binary


def _looks_like_text(crop: np.ndarray) -> bool:
    """Text is many small, similarly-sized blobs sitting on a few baselines;
    a diagram is a handful of large connected structures."""
    binary = _ink_mask(crop)
    h, w = binary.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return False

    boxes = stats[1:]  # drop background
    heights = boxes[:, cv2.CC_STAT_HEIGHT]
    areas = boxes[:, cv2.CC_STAT_AREA]

    glyphs = heights[(heights > 2) & (heights < 0.18 * h)]
    if len(glyphs) < 12:
        return False

    # Overwhelmingly small components, none of them structurally large.
    small_ratio = len(glyphs) / max(len(boxes), 1)
    largest_share = areas.max() / float(h * w)
    height_spread = float(np.std(glyphs)) / max(float(np.mean(glyphs)), 1e-6)

    return small_ratio > 0.8 and largest_share < 0.10 and height_spread < 0.6


def heuristic_verdict(crop: np.ndarray) -> tuple:
    """Returns (verdict, reason) where verdict is 'reject' | 'accept' | 'unsure'."""
    if crop is None or crop.size == 0:
        return "reject", "empty crop"

    h, w = crop.shape[:2]
    if h < MIN_SIDE_PX or w < MIN_SIDE_PX:
        return "reject", f"too small ({w}x{h}) — icon or glyph"
    if h * w < MIN_AREA_PX:
        return "reject", f"area {h * w}px below diagram threshold"

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > MAX_ASPECT:
        return "reject", f"aspect {aspect:.1f}:1 — rule or text strip"

    ink = float(np.count_nonzero(_ink_mask(crop))) / float(h * w)
    if ink < MIN_INK_RATIO:
        return "reject", f"ink ratio {ink:.4f} — effectively blank"
    if ink > MAX_INK_RATIO:
        return "reject", f"ink ratio {ink:.2f} — solid block"

    if _looks_like_text(crop):
        return "reject", "component profile matches a text block"

    # Big and structured enough to be obvious; skip the vision call.
    if h * w > 160_000 and 0.02 < ink < 0.85:
        return "accept", "large structured region"

    return "unsure", "needs vision check"


def _vision_verdict(crop: np.ndarray) -> tuple:
    import base64

    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        return True, "encode failed — keeping"
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    prompt = (
        "Classify this cropped region from a machine maintenance manual.\n"
        "Return JSON with keys:\n"
        '  "is_diagram": true only if it is a technical illustration, schematic, exploded view, '
        'photo of equipment, or wiring/flow diagram that would help a technician.\n'
        '  "kind": one of "diagram", "text", "icon", "logo", "decoration", "blank".\n'
        '  "reason": under 12 words.\n'
        "Set is_diagram false for plain text/paragraphs/tables, company logos, small warning "
        "icons, page furniture, borders and blank areas."
    )
    try:
        raw = chat_json(MODEL_VISION, prompt, image_b64=b64, max_tokens=250, temperature=0.0)
        data = loads_tolerant(raw) if raw else None
        if isinstance(data, dict) and "is_diagram" in data:
            kind = str(data.get("kind", "")).strip().lower()
            keep = bool(data["is_diagram"])
            # The model routinely answers is_diagram=true while labelling the crop
            # "icon" or "logo". The specific label is the more reliable signal, so
            # a rejecting `kind` overrides the boolean.
            if kind in REJECT_KINDS:
                keep = False
            return keep, f"{kind or '?'}: {data.get('reason', '')}"
    except Exception as e:
        print(f"      ⚠️ [Validator] Vision check failed: {e}")
    # Never drop a figure because the validator itself broke.
    return True, "validator unavailable — keeping"


def is_valid_figure(crop: np.ndarray, use_vision: bool = True) -> tuple:
    """(keep: bool, reason: str)."""
    verdict, reason = heuristic_verdict(crop)
    if verdict == "reject":
        return False, reason
    if verdict == "accept":
        return True, reason
    if not use_vision:
        return True, "unsure, vision disabled — keeping"
    return _vision_verdict(crop)
