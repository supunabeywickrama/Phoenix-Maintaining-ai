"""
Callout labels on scanned drawings: circled numbers ("(87)") and letter-number
item codes ("B08-13").

Plain OCR over a whole scanned drawing is nearly blind to circled numbers - on a
real scanned hydraulic drawing it found 3 of 36, because the ring around each
number reads as part of the character. So circles are found first by shape,
and only the digits inside each one are recognised, with everything outside the
circle blanked. Measured on the same drawing: 36 of 36 in 1.4 s.

Letter-number item codes have no circle; they are matched from OCR text with a
code pattern.
"""
import re

import cv2
import fitz  # PyMuPDF
import numpy as np

MIN_RADIUS, MAX_RADIUS = 9, 70         # px at the page image's native resolution
MIN_CIRCULARITY = 0.75
MIN_FILL = 0.7
RADIUS_TOLERANCE = 0.35                # callout circles on one drawing are one size
MIN_CONF = 0.5

ITEM_CODE = re.compile(r"^[A-Z]{1,2}\d{1,3}-\d{1,3}$")


def page_image(page: fitz.Page) -> np.ndarray:
    """A scanned page at its embedded image's own resolution."""
    info = page.get_image_info()
    native = max((max(i.get("width", 0), i.get("height", 0)) for i in info), default=0)
    zoom = native / max(page.rect.width, page.rect.height) if native else 300 / 72
    zoom = min(max(zoom, 2.0), 6.0)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _reader():
    import services.parts_list as P
    if P._ocr_reader is None:
        import easyocr
        P._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return P._ocr_reader


def find_circles(gray: np.ndarray) -> list:
    """(cx, cy, r) for every closed, round outline of callout size."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    contours, _ = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 120:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (MIN_RADIUS <= r <= MAX_RADIUS):
            continue
        per = cv2.arcLength(c, True)
        if not per:
            continue
        if 4 * np.pi * area / (per * per) > MIN_CIRCULARITY and area / (np.pi * r * r) > MIN_FILL:
            cands.append((int(cx), int(cy), int(r)))
    # A ring gives an outer and an inner contour - keep the outer one.
    cands.sort(key=lambda c: -c[2])
    circles = []
    for c in cands:
        if all((c[0] - o[0]) ** 2 + (c[1] - o[1]) ** 2 > (0.6 * o[2]) ** 2 for o in circles):
            circles.append(c)
    return circles


def circled_numbers(img: np.ndarray, exclude: list = ()) -> list:
    """[(number, cx, cy, r, confidence)] for circled callout numbers.

    `exclude`: (x, y, w, h) boxes to ignore, e.g. tables on the same page.
    A circle's number is kept only when the circle is the usual callout size
    for this drawing: port symbols and holes (a digit-like mark in a small or
    large ring) were read as "0" and "6" on the test drawing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    reader = _reader()
    reads = []
    for cx, cy, r in find_circles(gray):
        if any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in exclude):
            continue
        rr = int(r * 0.78)                       # inside the ring
        y0, x0 = max(0, cy - rr), max(0, cx - rr)
        crop = gray[y0:cy + rr, x0:cx + rr].copy()
        if crop.size == 0:
            continue
        mask = np.zeros_like(crop)
        cv2.circle(mask, (cx - x0, cy - y0), rr, 255, -1)
        crop[mask == 0] = 255
        if (crop < 128).mean() < 0.03:
            continue                              # empty ring
        big = np.pad(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC), 20, constant_values=255)
        best = max(reader.recognize(big, allowlist="0123456789", detail=1), key=lambda f: f[2], default=None)
        text = best[1].strip() if best else ""
        if text.isdigit() and best[2] >= MIN_CONF and 0 < int(text) < 1000:
            reads.append((int(text), cx, cy, r, float(best[2])))
    if not reads:
        return []
    # Callout circles on one drawing share a size; take the radius most numbers
    # were read at and drop circles far from it.
    typical = float(np.median([r for *_, r, _c in reads]))
    return [x for x in reads if abs(x[3] - typical) <= RADIUS_TOLERANCE * typical]


def item_codes(img: np.ndarray, exclude: list = ()) -> list:
    """[(code, cx, cy, confidence)] for letter-number labels like "B08-13"."""
    reader = _reader()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = []
    for box, text, conf in reader.readtext(gray, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"):
        code = text.strip().upper().replace("--", "-")
        m = re.match(r"^([A-Z]{1,2})(.*)$", code)
        if m:
            # Letters OCR confuses inside the number part: "BO8-13", "BI1-05".
            code = m.group(1) + m.group(2).translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
        if not ITEM_CODE.match(code) or conf < 0.4:
            continue
        cx = int(sum(p[0] for p in box) / 4)
        cy = int(sum(p[1] for p in box) / 4)
        if any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in exclude):
            continue
        out.append((code, cx, cy, float(conf)))
    return out
