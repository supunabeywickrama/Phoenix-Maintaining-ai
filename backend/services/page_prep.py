"""
Page preparation for scanned and photographed pages - run once, before any
other step reads the PDF.

A real manual PDF mixes digital pages with scans and phone photos. Measured on
one (JON WAI injection moulding machine, 873-style parts pages):
- every scanned page was rotated a quarter turn; read as-is, OCR found 0 words;
- phone photos include the blue folder behind the page and a slight perspective;
- scans carry oversized page dimensions (up to 2538x3752 pt), which throws off
  every size-based measurement downstream.

Each scanned page is therefore rebuilt as an upright, straightened, cropped
image page at a standard size. Digital pages (with a text layer) are copied
untouched. Page numbers stay 1:1 with the original PDF.
"""
import os
import tempfile
from dataclasses import dataclass, field

import cv2
import fitz  # PyMuPDF
import numpy as np

ANALYSIS_LONG_SIDE = 1600      # px - orientation check resolution
OUTPUT_LONG_SIDE = 4000        # px - cap on the rebuilt page image
PAGE_LONG_SIDE_PT = 842.0      # rebuilt pages use A4's long side (points)
CONFIDENT_WORDS = 25           # upright text found at 0 deg -> accept without trying others
TIE_MARGIN = 0.85              # runner-up within 15% of the best -> ask vision to break the tie
MAX_SKEW_DEG = 8.0

ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class PageFix:
    page: int
    scanned: bool
    rotation: int = 0
    cropped: bool = False
    skew: float = 0.0
    notes: list = field(default_factory=list)


def is_scanned(page: fitz.Page) -> bool:
    """No usable text layer and images covering most of the page."""
    if len(page.get_text("words")) >= 5:
        return False
    info = page.get_image_info()
    covered = sum(fitz.Rect(i["bbox"]).get_area() for i in info)
    return covered / max(page.rect.get_area(), 1) > 0.5


def _render(page: fitz.Page) -> np.ndarray:
    """The page as pixels at close to its embedded image resolution."""
    long_pt = max(page.rect.width, page.rect.height)
    best = max((i.get("width", 0), i.get("height", 0)) for i in page.get_image_info()) if page.get_image_info() else (0, 0)
    native = max(best) if best else 0
    zoom = min(max(native / long_pt, 1.0) if native else 2.0, OUTPUT_LONG_SIDE / long_pt)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ── Crop a photo to the sheet of paper ───────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def crop_to_sheet(img: np.ndarray) -> tuple:
    """(image, cropped?) - find the bright sheet of paper against its background
    and flatten its perspective. Left alone when no clear sheet outline is
    found or it already fills the image."""
    small = _resize_long(img, 1000)
    scale = img.shape[1] / small.shape[1]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Paper is the large bright region; background (desk, folder) is darker or coloured.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    paper = ((hsv[:, :, 1] < 60) & (gray > 150)).astype(np.uint8) * 255
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not (0.35 < area < 0.93):
        return img, False
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(approx) != 4:
        x, y, w, h = cv2.boundingRect(c)
        return img[int(y * scale):int((y + h) * scale), int(x * scale):int((x + w) * scale)], True
    src = _order_corners(approx.reshape(4, 2).astype(np.float32) * scale)
    width = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
    height = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (width, height),
                               borderValue=(255, 255, 255)), True


# ── Rotation ─────────────────────────────────────────────────────────────────

def _ocr_reader():
    from services import parts_list
    if parts_list._ocr_reader is None:
        import easyocr
        parts_list._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return parts_list._ocr_reader


def _upright_words(img: np.ndarray) -> int:
    res = _ocr_reader().readtext(img, text_threshold=0.6)
    return sum(1 for _, t, c in res if len(t.strip()) >= 3 and c > 0.5)


def _vision_is_upright(img: np.ndarray) -> bool:
    import base64
    from unified_rag.ai_client import chat_json, MODEL_VISION
    from services.llm_json import loads_tolerant
    ok, buf = cv2.imencode(".jpg", _resize_long(img, 1200))
    try:
        data = loads_tolerant(chat_json(
            MODEL_VISION,
            'Is the printed text in this image upright and readable left-to-right (not sideways or '
            'upside down)? Return JSON {"upright": true or false}',
            image_b64=base64.b64encode(buf.tobytes()).decode(), max_tokens=40, temperature=0.0) or "")
        return bool(isinstance(data, dict) and data.get("upright"))
    except Exception:
        return False


def detect_rotation(img: np.ndarray) -> tuple:
    """(degrees clockwise to rotate, note). Tested on 6 real scanned/photographed
    pages: upright text found 0-2 words at the wrong rotation and 20-90 at the
    right one; one page tied closely (32 vs 30), hence the vision tie-break."""
    small = _resize_long(img, ANALYSIS_LONG_SIDE)
    first = _upright_words(small)
    if first >= CONFIDENT_WORDS:
        return 0, f"upright ({first} words)"
    scores = {0: first}
    for k in (90, 270, 180):
        scores[k] = _upright_words(cv2.rotate(small, ROTATIONS[k]))
    if max(scores.values()) < 8:
        # Dense drawings (a hydraulic schematic) have text too small to read at
        # the check resolution - it found 1 word in every rotation and the page
        # was left sideways. Check again at double resolution.
        big = _resize_long(img, ANALYSIS_LONG_SIDE * 2)
        scores = {k: _upright_words(big if ROTATIONS[k] is None else cv2.rotate(big, ROTATIONS[k]))
                  for k in (0, 90, 270, 180)}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[0], ranked[1]
    if scores[best] < 3:
        for k in (0, 90, 270, 180):
            im = small if ROTATIONS[k] is None else cv2.rotate(small, ROTATIONS[k])
            if _vision_is_upright(im):
                return k, f"rotation {k} (no readable text {scores}; decided by vision)"
        return 0, f"no readable text in any rotation {scores} - left as is"
    if scores[second] >= scores[best] * TIE_MARGIN:
        for k in (best, second):
            im = small if ROTATIONS[k] is None else cv2.rotate(small, ROTATIONS[k])
            if _vision_is_upright(im):
                return k, f"rotation {k} (OCR tie {scores}, confirmed by vision)"
        return best, f"rotation {best} (OCR tie {scores}, vision undecided)"
    return best, f"rotation {best} (OCR words by rotation {scores})"


# ── Skew ─────────────────────────────────────────────────────────────────────

def detect_skew(img: np.ndarray) -> float:
    """Small tilt in degrees from long near-horizontal lines (table rules, frames,
    title blocks). A tilted parts list makes rows drift across the page - measured
    ~40 px over a 2679 px table - and row reading jumbles neighbouring rows."""
    small = _resize_long(img, 2000)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=200,
                            minLineLength=small.shape[1] // 4, maxLineGap=10)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= MAX_SKEW_DEG:
            angles.append(a)
    return float(np.median(angles)) if len(angles) >= 3 else 0.0


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


# ── Whole document ───────────────────────────────────────────────────────────

def prepare_pdf(path: str) -> tuple:
    """(path to use, [PageFix]). Returns the original path unchanged when the PDF
    has no scanned pages."""
    src = fitz.open(path)
    if not any(is_scanned(p) for p in src):
        return path, [PageFix(page=i + 1, scanned=False) for i in range(len(src))]

    out = fitz.open()
    fixes = []
    for i, page in enumerate(src):
        if not is_scanned(page):
            out.insert_pdf(src, from_page=i, to_page=i)
            fixes.append(PageFix(page=i + 1, scanned=False))
            continue
        fix = PageFix(page=i + 1, scanned=True)
        img = _render(page)
        img, fix.cropped = crop_to_sheet(img)
        rot, note = detect_rotation(img)
        fix.rotation, fix.notes = rot, [note]
        if ROTATIONS[rot] is not None:
            img = cv2.rotate(img, ROTATIONS[rot])
        fix.skew = round(detect_skew(img), 2)
        img = deskew(img, fix.skew)
        img = _resize_long(img, OUTPUT_LONG_SIDE)

        h, w = img.shape[:2]
        scale = PAGE_LONG_SIDE_PT / max(h, w)
        new = out.new_page(width=w * scale, height=h * scale)
        ok, buf = cv2.imencode(".png", img)
        new.insert_image(new.rect, stream=buf.tobytes())
        fixes.append(fix)
        print(f"   [PagePrep] Page {i + 1}: {note}; cropped to sheet: {fix.cropped}; "
              f"skew corrected {fix.skew} deg")

    fd, prepared = tempfile.mkstemp(suffix="_prepared.pdf")
    os.close(fd)
    out.save(prepared, garbage=3, deflate=True)
    return prepared, fixes
