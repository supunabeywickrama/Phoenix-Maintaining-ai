"""
Parts lists — the "Ref. No. / Part Number / Description / Qty." tables that
decode the numbered callouts on an exploded-view figure.

In a parts manual the figure and its list are usually separate: the drawing
fills page N and the list is on page N+1 under the same subtitle ("EUROPEAN
ACCESSORIES (Cont'd) / (Caliper Brake Kit)"); smaller kits put both on one page.

HOW ROWS ARE READ
Every reader produces positioned words, and one shared step rebuilds the rows
from those positions against the table's own header columns.

- Digital PDF: the words are the PDF's printed text. Exact characters, exact
  rows - verified.
- Scanned page: the words come from OCR, which places every word correctly but
  misreads characters (tested: ref "1" read as "7", "O-RING" as "0-RING",
  small quantities not detected). So OCR rows are then checked two ways:
    1. Refs must count up (1, 2, 3...): a "7" between nothing and "2" is 1.
    2. A separate vision read of the page is matched to each OCR row BY PART
       NUMBER, never by position. Vision alone misaligns rows on dense lists -
       tested on the Caliper Brake Kit list it merged "97K-4 O-RING" into ref 5
       and shifted every later ref by one - but a part number and the
       description beside it do come back together. Agreement confirms a row;
       disagreement is kept visible on the row rather than hidden.
"""
import base64
import re
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
import numpy as np

from unified_rag.ai_client import chat_json, MODEL_VISION
from services.llm_json import loads_tolerant

TEXT_LINE_TOLERANCE = 3.0   # points: printed words this close vertically share a row
OCR_LINE_TOLERANCE = 5.0    # OCR boxes wobble a little more
COLUMN_SLACK = 4.0          # points: a value may start slightly left of its header
MAX_ROW_GAP = 40.0          # points: a gap this large below the last row ends the table
MAX_REF = 999
OCR_DPI = 300
OCR_MIN_CONF_CALLOUT = 0.5


def clean(text: str) -> str:
    """Printed-text normalisation. This manual's font encodes a hyphen as '--'
    in the text layer ('83F--3' is printed '83F-3') and uses curly quotes."""
    text = (text or "").replace("--", "-").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class PartRow:
    ref: str
    part_number: str
    description: str
    remarks: str = ""
    qty: str = ""
    # How far this row can be trusted: "printed text", "ocr+vision agree",
    # or a sentence saying what could not be confirmed.
    check: str = "printed text"


@dataclass
class PartsList:
    page: int                      # 1-based page the list starts on
    title: str
    subtitle: str
    rows: list
    source: str                    # "text_layer" | "ocr+vision" | "vision"
    verified: bool                 # True only when read from the printed text
    pages: list = field(default_factory=list)

    @property
    def refs(self) -> set:
        return {r.ref for r in self.rows if r.ref}

    def grouped(self) -> list:
        """[(ref, [rows])] — a row printed with no ref number is an alternative
        part for the ref above it (e.g. 83FN-3 under ref 2's 83F-3)."""
        groups, current = [], None
        for row in self.rows:
            if row.ref or current is None:
                current = (row.ref, [row])
                groups.append(current)
            else:
                current[1].append(row)
        return groups

    def to_markdown(self) -> str:
        scanned = not self.verified
        head = "| Ref. | Part Number | Description | Remarks | Qty. |" + (" Reading |" if scanned else "")
        sep = "| --- | --- | --- | --- | --- |" + (" --- |" if scanned else "")
        out = [head, sep]
        for r in self.rows:
            cells = [r.ref, r.part_number, r.description, r.remarks, r.qty] + ([r.check] if scanned else [])
            out.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
        return "\n".join(out)

    def to_text(self) -> str:
        """Readable, embeddable form: every ref with its part and quantity."""
        head = " ".join(filter(None, [self.title, self.subtitle])) or "Parts list"
        lines = [f"{head} — parts list"]
        for ref, rows in self.grouped():
            first, alts = rows[0], rows[1:]
            line = f"Ref {ref or '-'}: {first.description} — part {first.part_number}"
            if first.qty:
                line += f" (qty {first.qty})"
            if first.remarks:
                line += f" — {first.remarks}"
            for a in alts:
                line += f"; alternative {a.part_number} {a.description}".rstrip()
                if a.qty:
                    line += f" (qty {a.qty})"
                if a.remarks:
                    line += f" — {a.remarks}"
            lines.append(line)
        return "\n".join(lines)


# ── Words: printed text or OCR ───────────────────────────────────────────────

# (x0, y0, x1, y1, text, confidence), in PDF points.
_ocr_cache = {}
_ocr_reader = None


def page_has_text_layer(page: fitz.Page) -> bool:
    return len(page.get_text("words")) >= 5


def _text_layer_words(page: fitz.Page) -> list:
    return [(w[0], w[1], w[2], w[3], w[4], 1.0) for w in page.get_text("words")]


def ocr_words(page: fitz.Page) -> list:
    """OCR a page into positioned words. Multi-word OCR boxes are split into
    words with positions spread by character count, so each word lands in its
    own column. Cached per page: a scanned figure page is read for its callout
    numbers and again when pairing."""
    key = (page.parent.name, page.number)
    if key in _ocr_cache:
        return _ocr_cache[key]
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    scale = page.rect.width / pix.width
    words = []
    for box, text, conf in _ocr_reader.readtext(img):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1, y0, y1 = min(xs) * scale, max(xs) * scale, min(ys) * scale, max(ys) * scale
        text = text.strip()
        if not text:
            continue
        span = max(len(text), 1)
        for m in re.finditer(r"\S+", text):
            wx0 = x0 + (x1 - x0) * m.start() / span
            wx1 = x0 + (x1 - x0) * m.end() / span
            words.append((wx0, y0, wx1, y1, m.group(0), float(conf)))
    _ocr_cache[key] = words
    return words


def page_words(page: fitz.Page) -> list:
    """Printed text when the page has it, OCR when it is a scan."""
    return _text_layer_words(page) if page_has_text_layer(page) else ocr_words(page)


def _lines(words: list, tolerance: float) -> list:
    """Words grouped into printed lines: [(y_center, [(x0, x1, text, conf), ...])]."""
    ordered = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, current, cy = [], [], None
    for w in ordered:
        yc = (w[1] + w[3]) / 2
        if cy is not None and abs(yc - cy) > tolerance:
            lines.append((cy, sorted(current)))
            current = []
        if not current:
            cy = yc
        current.append((w[0], w[2], w[4], w[5]))
    if current:
        lines.append((cy, sorted(current)))
    return lines


# ── Rows from positioned words ───────────────────────────────────────────────

def _fold(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _header(lines: list) -> Optional[tuple]:
    """(index of the header's last line, column starts, header top y).

    Tolerant of OCR ("Ref:", "No:", "Oty:"/"Aty:" for "Qty.") and of the header
    being split over two lines ("Ref. Part" above "No. Number Description").
    """
    for i, (y, words) in enumerate(lines):
        if not any(_fold(t) == "description" for _, _, t, _ in words):
            continue
        # The header can start a line above "Description" ("Ref. Part") and OCR
        # can drop "Qty." a few points lower - but it never reaches the next
        # row, 13pt down. A wider window swallowed ref 1 on the Caliper Brake
        # Kit list, whose remarks read "Ref. 2 - 32".
        near = [(x0, t, ly) for ly, ws in lines if -14 <= ly - y <= 6 for x0, _, t, _ in ws]
        desc_x = next(x0 for x0, t, _ in near if _fold(t) == "description")
        qty_x = next((x0 for x0, t, _ in near if re.fullmatch(r"[qoa0]t[yv]", _fold(t))), None)
        if qty_x is None:
            continue
        ref_x = min((x0 for x0, t, _ in near if _fold(t) in ("ref", "no") and x0 < desc_x), default=0.0)
        part_x = min((x0 for x0, t, _ in near if _fold(t) in ("part", "number") and ref_x < x0 < desc_x),
                     default=None)
        if part_x is None:
            continue
        cols = {"ref": ref_x, "part": part_x, "description": desc_x, "qty": qty_x}
        remarks = next((x0 for x0, t, _ in near if _fold(t) == "remarks"), None)
        serial = next((x0 for x0, t, _ in near if _fold(t) == "serial"), None)
        if remarks is not None:
            cols["remarks"] = remarks
        if serial is not None:
            cols["serial"] = serial
        header_ys = [ly for _, t, ly in near if _fold(t) in
                     ("ref", "no", "part", "number", "description", "remarks", "serial")
                     or re.fullmatch(r"[qoa0]t[yv]", _fold(t))]
        last = max(j for j, (ly, _) in enumerate(lines) if ly <= max(header_ys) + 0.01)
        return last, cols, min(header_ys)
    return None


def _column(x0: float, cols: dict) -> str:
    best, best_x = "ref", -1.0
    for name, start in cols.items():
        if start <= x0 + COLUMN_SLACK and start > best_x:
            best, best_x = name, start
    return best


def _title_above(lines: list, header_top: float) -> tuple:
    """(title, subtitle) printed just above the header, skipping NOTE lines and
    drawing codes."""
    picked = []
    for y, words in reversed([ln for ln in lines if ln[0] < header_top - 2]):
        if header_top - y > 70:
            break
        text = clean(" ".join(t for _, _, t, _ in words))
        if not text or text.upper().startswith("NOTE") or re.fullmatch(r"[A-Z]{1,3}-\d+", text):
            continue
        picked.insert(0, text)
        if len(picked) == 2:
            break
    subtitle = next((t for t in picked if t.startswith("(")), "")
    title = next((t for t in picked if not t.startswith("(")), "")
    return title, subtitle


def _rows_from_words(words: list, page: fitz.Page, tolerance: float, ocr: bool) -> Optional[tuple]:
    """(title, subtitle, rows, ref_confidences, row_y_centers, columns) or None if
    the page has no parts-list header."""
    lines = _lines(words, tolerance)
    found = _header(lines)
    if not found:
        return None
    header_index, cols, header_top = found
    title, subtitle = _title_above(lines, header_top)

    rows, confs, row_ys, row_lines, last_y = [], [], [], [], lines[header_index][0]
    footer_y = page.rect.height * 0.93
    for y, line_words in lines[header_index + 1:]:
        if y > footer_y or y - last_y > MAX_ROW_GAP:
            break
        cells = {k: [] for k in ("ref", "part", "description", "remarks", "serial", "qty")}
        cell_conf = {k: 1.0 for k in cells}
        for x0, _, text, conf in line_words:
            col = _column(x0, cols)
            cells[col].append(text)
            cell_conf[col] = min(cell_conf[col], conf)
        cell = {k: clean(" ".join(v)) for k, v in cells.items()}
        if ocr:
            # Characters OCR confuses in exactly these places.
            cell["part"] = re.sub(r"\s*-\s*", "-", cell["part"])
            cell["description"] = re.sub(r"\b0(?=-?[A-Z]{2})", "O", cell["description"])
            cell["remarks"] = re.sub(r"\bWIO\b", "W/O", cell["remarks"])
            cell["ref"] = cell["ref"].strip(".:;, ")
            cell["qty"] = cell["qty"].strip(".:;, ")

        # Refs can carry a printed marker for a changed part ("4*", "23*"), and
        # an alternative row may show the marker alone ("*").
        m = re.fullmatch(r"(\d{1,3})?\s*(\*)?", cell["ref"] or "")
        ref = (m.group(1) or "") if m else ""
        marked = bool(m and m.group(2))
        if ref and not (0 < int(ref) <= MAX_REF):
            ref = ""
        last_y = y

        if not ref and not cell["part"] and not cell["description"]:
            # Wrapped remarks - continue the row above.
            if rows:
                prev = rows[-1]
                row_lines[-1].append(y)
                prev.remarks = clean(f"{prev.remarks} {cell['remarks']} {cell['serial']}")
                if not prev.qty and cell["qty"]:
                    prev.qty = cell["qty"]
            continue
        remarks = clean(" ".join(filter(None, [cell["remarks"], cell["serial"],
                                               "(marked * in the parts list)" if marked else ""])))
        rows.append(PartRow(ref=ref, part_number=cell["part"], description=cell["description"],
                            remarks=remarks, qty=cell["qty"],
                            check="printed text" if not ocr else "ocr only"))
        confs.append(cell_conf["ref"])
        row_ys.append(y)
        row_lines.append([y])
    if not rows:
        return None
    return title, subtitle, rows, confs, row_ys, cols, row_lines


def read_text_layer(page: fitz.Page) -> Optional[PartsList]:
    """Parse a parts list from the page's printed text, or None if the page has
    no parts-list header."""
    got = _rows_from_words(_text_layer_words(page), page, TEXT_LINE_TOLERANCE, ocr=False)
    if not got:
        return None
    title, subtitle, rows, *_ = got
    pno = page.number + 1
    return PartsList(page=pno, title=_strip_contd(title), subtitle=subtitle, rows=rows,
                     source="text_layer", verified=True, pages=[pno])


def _strip_contd(title: str) -> str:
    return re.sub(r"\s*\(Cont'?d\)\s*", " ", title or "", flags=re.I).strip()


# ── Scanned pages: OCR rows, checked by sequence and by a vision read ───────

QTY_PATTERN = re.compile(r"^(\d{1,3}|[xX]|A/R|AR)$")


def _part_key(part_number: str) -> str:
    """Part number folded for comparison only: OCR/vision confuse O/0 and I/1."""
    return re.sub(r"[^A-Z0-9]", "", (part_number or "").upper()).replace("O", "0").replace("I", "1")


def _desc_words(text: str) -> set:
    return {w.replace("0", "o") for w in re.findall(r"[a-z0]{3,}", (text or "").lower())}


def read_with_vision(page: fitz.Page, dpi: int = 150) -> Optional[PartsList]:
    """A vision model's read of a parts-list page. Used only as the second
    reader for a scanned page - never trusted on its own for row alignment."""
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    prompt = (
        "Is this page a parts list table (columns like Ref. No., Part Number, Description, "
        "Qty.)? If yes, transcribe EVERY row exactly as printed.\n"
        'Return JSON: {"is_parts_list": bool, "title": str, "subtitle": str, "rows": '
        '[{"ref": str, "part_number": str, "description": str, "remarks": str, "qty": str}]}\n'
        "One object per printed line, top to bottom. A line with a blank Ref. No. keeps ref \"\". "
        "Copy part numbers character by character. Never merge two printed lines into one row."
    )
    try:
        data = loads_tolerant(chat_json(MODEL_VISION, prompt,
                                        image_b64=base64.b64encode(png).decode("utf-8"),
                                        max_tokens=4000, temperature=0.0) or "")
    except Exception as e:
        print(f"      ⚠️ [PartsList] Vision read failed on page {page.number + 1}: {e}")
        return None
    if not isinstance(data, dict) or not data.get("is_parts_list"):
        return None
    rows = []
    for r in data.get("rows") or []:
        if not isinstance(r, dict):
            continue
        ref = clean(str(r.get("ref", "")))
        ref = ref if re.fullmatch(r"\d{1,3}", ref) else ""
        part, desc = clean(str(r.get("part_number", ""))), clean(str(r.get("description", "")))
        if part or desc:
            rows.append(PartRow(ref, part, desc, clean(str(r.get("remarks", ""))),
                                clean(str(r.get("qty", ""))), check="vision only"))
    if not rows:
        return None
    pno = page.number + 1
    return PartsList(page=pno, title=_strip_contd(clean(str(data.get("title", "")))),
                     subtitle=clean(str(data.get("subtitle", ""))), rows=rows,
                     source="vision", verified=False, pages=[pno])


COLUMN_OCR_DPI = 500
COLUMN_MATCH_POINTS = 4.5


def _ocr_strip(page: fitz.Page, rect: fitz.Rect, allowlist: str) -> list:
    """OCR one narrow column at high resolution, restricted to the characters
    that column can contain. [(y_center_points, text, confidence)]."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    clip = rect & page.rect
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        return []
    pix = page.get_pixmap(clip=clip, dpi=COLUMN_OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
    scale = clip.height / pix.height
    out = []
    for box, text, conf in _ocr_reader.readtext(img, allowlist=allowlist):
        ys = [p[1] for p in box]
        out.append((clip.y0 + (min(ys) + max(ys)) / 2 * scale, text.strip(), float(conf)))
    return out


CELL_HALF_HEIGHT = 6.0      # points above/below a row's centre line
CELL_MIN_CONF = 0.4


def _read_cells(page: fitz.Page, x0: float, x1: float, row_ys: list, allowlist: str) -> list:
    """Recognise one narrow column cell by cell: [(text, confidence) or None].

    Recognition only, no text detection. Whole-column OCR was tried first and
    failed: a column crop is a tall thin image (608x4617 px for one list),
    which the detector shrinks until lone small digits vanish - it found 6 of
    44 quantities. Each cell is instead cut out and read directly.
    """
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    top, bottom = min(row_ys) - CELL_HALF_HEIGHT - 2, max(row_ys) + CELL_HALF_HEIGHT + 2
    clip = fitz.Rect(x0, top, x1, bottom) & page.rect
    if clip.is_empty:
        return [None] * len(row_ys)
    pix = page.get_pixmap(clip=clip, dpi=COLUMN_OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    # Erase table rulings. With only digits allowed, a vertical column rule at
    # the edge of the crop was read as "4" - every "1" came back as "41".
    # Only LONG unbroken vertical strokes are removed (taller than several
    # rows); digits stacked in a column are broken by the white gap between
    # rows and survive. Two simpler rules failed on real scans: erasing any
    # mostly-dark pixel column also erased stacked "1"s, and a narrower one
    # left a grey sliver that read as a leading "1" ("14" for 4). The line mask
    # is widened a few pixels to take the anti-aliased edge with it.
    import cv2
    ink = (img.min(axis=2) < 180).astype(np.uint8)
    run = max(40, int(COLUMN_OCR_DPI / 72 * 3 * CELL_HALF_HEIGHT))   # ~3 rows tall, in pixels
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)))
    # Then keep only the band BETWEEN rulings. Erasing alone was not enough:
    # tested cell by cell, a crop that still touched the rule read "4" as
    # "14", "2" as "12" and "x" as "7x".
    width = img.shape[1]
    line_cols = np.where(lines.any(axis=0))[0]
    left = max((c for c in line_cols if c < width * 0.45), default=-1) + 8
    right = min((c for c in line_cols if c > width * 0.55), default=width) - 8
    if right - left > 12:
        img = img[:, left:right]
    px_per_pt = pix.height / clip.height
    out = []
    for y in row_ys:
        a = max(0, int((y - CELL_HALF_HEIGHT - clip.y0) * px_per_pt))
        b = min(pix.height, int((y + CELL_HALF_HEIGHT - clip.y0) * px_per_pt))
        cell = np.ascontiguousarray(img[a:b])
        # Horizontal rule through the cell (the line under the header, or between
        # row groups), with its grey anti-aliased edge: left in, it read as a
        # leading "7" on the first and last rows ("71" for 1).
        rule_rows = (cell.min(axis=2) < 180).mean(axis=1) > 0.4
        rule_rows = np.convolve(rule_rows.astype(int), np.ones(5, dtype=int), mode="same") > 0
        cell[rule_rows] = 255
        # A blank cell must stay blank: skip crops with (almost) no ink.
        if cell.size == 0 or (cell.min(axis=2) < 128).mean() < 0.002:
            out.append(None)
            continue
        # White margin round the digit: measured confidence 0.75 -> 0.98 on the same cell.
        cell = np.pad(np.ascontiguousarray(cell.min(axis=2)), 20, constant_values=255)
        found = _ocr_reader.recognize(cell, allowlist=allowlist, detail=1)
        best = max(found, key=lambda f: f[2], default=None)
        text = (best[1] if best else "").strip()
        out.append((text, float(best[2])) if best and text and best[2] >= CELL_MIN_CONF else None)
    return out


def _reread_narrow_columns(page: fitz.Page, rows: list, confs: list, row_ys: list, cols: dict,
                           row_lines: Optional[list] = None) -> None:
    """Re-read the Ref and Qty columns cell by cell.

    Whole-page OCR misses exactly these: a lone small digit in a wide empty
    column. Measured on a scanned Caliper Brake Kit list it dropped ref "7"
    (so a TUBE looked like an alternative of ref 6) and read almost no
    quantities - which left nothing confirmable.
    """
    if not row_ys:
        return
    ref_cells = _read_cells(page, cols["ref"] - 12, cols["part"] - 2, row_ys, "0123456789*")
    # Just the Qty column: a quantity is at most three characters, so the crop
    # stops before the table's right-hand border.
    # A wrapped row prints its quantity on its LAST line ("Was 6725360 [A]  1"),
    # so every line of the row is read and the first valid quantity is used.
    row_lines = row_lines or [[y] for y in row_ys]
    flat = [y for ys in row_lines for y in ys]
    flat_cells = _read_cells(page, cols["qty"] - 3, cols["qty"] + 24, flat, "0123456789xX")
    qty_cells, k = [], 0
    for ys in row_lines:
        cells = flat_cells[k:k + len(ys)]
        k += len(ys)
        qty_cells.append(next((c for c in cells if c and QTY_PATTERN.match(c[0])), None))
    for i in range(len(rows)):
        if ref_cells[i]:
            text, conf = ref_cells[i][0].replace("*", "").strip(), ref_cells[i][1]
            if re.fullmatch(r"\d{1,3}", text) and 0 < int(text) <= MAX_REF:
                if not rows[i].ref or (rows[i].ref != text and conf > confs[i]):
                    rows[i].ref, confs[i] = text, conf
                elif rows[i].ref == text:
                    confs[i] = max(confs[i], conf)
        if qty_cells[i] and QTY_PATTERN.match(qty_cells[i][0]):
            rows[i].qty = qty_cells[i][0]


def read_scanned(page: fitz.Page) -> Optional[PartsList]:
    """Parts list from a scanned page: OCR rows, checked against an independent
    vision read and against the ref count.

    A row is marked confirmed only when its ref, part number, description AND
    quantity are all established. Anything short of that is kept but says
    exactly what is unconfirmed. Measured against the printed text of two real
    lists, an earlier version confirmed a TUBE under the wrong ref (OCR missed
    the small "7", so the row looked like an alternative of ref 6) and
    confirmed quantities OCR had read as "[" and "|".
    """
    got = _rows_from_words(ocr_words(page), page, OCR_LINE_TOLERANCE, ocr=True)
    if not got:
        return None
    title, subtitle, rows, confs, row_ys, cols, row_lines = got
    _reread_narrow_columns(page, rows, confs, row_ys, cols, row_lines)
    vision = read_with_vision(page)

    # Vision rows by part number. Vision's row ORDER is not trusted, but a part
    # number and the description printed beside it do come back together. A
    # merged vision row ("97K-4 6710421" / "O-RING TUBE") lines its tokens up
    # in order, so each part number keeps its own words.
    by_part = {}
    for v in (vision.rows if vision else []):
        tokens = [t for t in re.split(r"[\s/]+", v.part_number) if _part_key(t)]
        descs = v.description.split()
        for n, t in enumerate(tokens):
            own_desc = descs[n] if len(tokens) == len(descs) and len(tokens) > 1 else v.description
            by_part.setdefault(_part_key(t), (v, t, own_desc, len(tokens) == 1))

    info = []
    for row, conf in zip(rows, confs):
        match = by_part.get(_part_key(row.part_number)) if row.part_number else None
        e = {"row": row, "conf": conf, "match": match, "notes": [],
             "name_ok": False, "ref_ok": False, "qty_ok": False}
        if match:
            v, v_part, v_desc, _single = match
            e["name_ok"] = bool(_desc_words(row.description) & _desc_words(v_desc)) or not row.description
            if e["name_ok"]:
                if not row.description:
                    row.description = v_desc
                if v_part != row.part_number:
                    # Same part number once O/0 and I/1 are folded: OCR confused
                    # the characters (e.g. "1OC416" for "10C-416"); vision keeps
                    # them in context, so its spelling is stored.
                    e["notes"].append(f"part number spelling from vision (OCR read {row.part_number})")
                    row.part_number = v_part
        info.append(e)

    # ── Refs: a missing ref is filled only from vision, only where it fits ──
    for i, e in enumerate(info):
        row, match = e["row"], e["match"]
        if row.ref:
            continue
        v_ref = match[0].ref if (match and e["name_ok"]) else None
        if v_ref == "":
            e["ref_ok"] = True
            e["notes"].append("alternative part (no ref on either reading)")
            continue
        above = next((x["row"].ref for x in reversed(info[:i]) if x["row"].ref), "")
        if v_ref and above and v_ref == above:
            # Vision lists this part under the same ref as the numbered row just
            # above it (typically by merging "SALES / SALES" into one row), so
            # both readings put it in that ref's group: an alternative part.
            e["ref_ok"] = True
            e["notes"].append(f"alternative part of ref {above}")
            continue
        if v_ref and v_ref.isdigit():
            prev = max((int(x["row"].ref) for x in info[:i] if x["row"].ref), default=0)
            later = [int(x["row"].ref) for x in info[i + 1:] if x["row"].ref and int(x["row"].ref) > prev]
            used = {x["row"].ref for x in info if x["row"].ref}
            if int(v_ref) > prev and (not later or int(v_ref) < later[0]) and v_ref not in used:
                row.ref = v_ref
                e["ref_ok"] = True
                e["from_vision"] = True
                e["notes"].append("ref from vision (OCR could not read it)")
                continue
        e["notes"].append("no ref read - may be its own ref or an alternative of the part above")

    # ── Refs: the count must go up by one ──
    numbered = [e for e in info if e["row"].ref]
    prev = 0   # last ref still standing - a cleared ref must not become the reference point
    for k, e in enumerate(numbered):
        if k:
            prev = next((int(x["row"].ref) for x in reversed(numbered[:k]) if x["row"].ref), 0)
        cur = int(e["row"].ref)
        nxt = int(numbered[k + 1]["row"].ref) if k + 1 < len(numbered) else None
        if cur == prev + 1 and (nxt is None or nxt > cur):
            e["ref_ok"] = e["ref_ok"] or e["conf"] >= 0.6
        elif nxt is not None and nxt == prev + 2 and cur != prev + 1:
            e["notes"].append(f"ref read as {cur}, corrected to {prev + 1} by the count")
            e["row"].ref = str(prev + 1)
            e["ref_ok"] = True
        elif cur <= prev or (nxt is not None and cur >= nxt):
            e["notes"].append(f"ref read as {cur} is out of order - cleared")
            e["row"].ref = ""
            e["ref_ok"] = False
        else:
            # A real gap in the count (3 then 5): keep the reading, but only
            # trust it when OCR was confident.
            e["ref_ok"] = e["ref_ok"] or e["conf"] >= 0.6

    # ── Quantities ──
    for e in info:
        row, match = e["row"], e["match"]
        ocr_qty = row.qty if QTY_PATTERN.match(row.qty or "") else ""
        v_qty = match[0].qty if (match and match[3] and QTY_PATTERN.match(match[0].qty or "")) else ""
        if ocr_qty and v_qty and ocr_qty.lower() == v_qty.lower():
            row.qty, e["qty_ok"] = ocr_qty, True
        elif ocr_qty and v_qty:
            row.qty = ocr_qty
            e["notes"].append(f"quantity uncertain: OCR read {ocr_qty}, vision read {v_qty}")
        elif v_qty:
            row.qty = v_qty
            e["notes"].append("quantity read by vision only")
        elif ocr_qty:
            row.qty = ocr_qty
            e["notes"].append("quantity read by OCR only")
        else:
            row.qty = ""
            e["notes"].append("quantity not readable")

    # ── Status per row ──
    stats = {"confirmed": 0, "unconfirmed": 0, "vision_only": 0}
    for e in info:
        row = e["row"]
        if not e["match"]:
            e["notes"].insert(0, "OCR only - vision did not read this part number")
        elif not e["name_ok"]:
            e["notes"].insert(0, f"description uncertain: OCR read '{row.description}', "
                                 f"vision read '{e['match'][2]}'")
        if e["match"] and e["name_ok"] and e["ref_ok"] and e["qty_ok"]:
            extra = [n for n in e["notes"] if not n.startswith("alternative part")]
            row.check = "confirmed by OCR and vision" + (f" ({'; '.join(extra)})" if extra else "")
            stats["confirmed"] += 1
        elif e["match"] and e["name_ok"] and e["ref_ok"]:
            # Ref, part number and description agree on both readings - the
            # callout-to-part mapping is established. Only the quantity is not.
            qty_notes = [n for n in e["notes"] if "quantity" in n]
            row.check = "part confirmed by OCR and vision; QUANTITY NOT CONFIRMED - " + (
                "; ".join(qty_notes) or "quantity unclear") + " - check against the manual"
            stats["part_confirmed"] = stats.get("part_confirmed", 0) + 1
        else:
            row.check = "UNCONFIRMED - " + "; ".join(e["notes"] or ["not confirmed"]) + " - check against the manual"
            stats["unconfirmed"] += 1

    # ── Rows only vision read: added so they are not lost, never confirmed ──
    seen = {_part_key(e["row"].part_number) for e in info if e["row"].part_number}
    refs_now = {e["row"].ref for e in info if e["row"].ref}
    for v in (vision.rows if vision else []):
        if not v.part_number or len(v.part_number.split()) > 1 or _part_key(v.part_number) in seen:
            continue
        placed = v.ref if (v.ref and v.ref not in refs_now) else ""
        new = PartRow(ref=placed, part_number=v.part_number, description=v.description,
                      remarks=v.remarks, qty=v.qty if QTY_PATTERN.match(v.qty or "") else "",
                      check="UNCONFIRMED - read by vision only (OCR could not read this row); its "
                            "ref and position are not confirmed - check against the manual")
        pos = next((j for j, r in enumerate(rows) if placed and r.ref and int(r.ref) > int(placed)), len(rows))
        rows.insert(pos, new)
        if placed:
            refs_now.add(placed)
        seen.add(_part_key(v.part_number))
        stats["vision_only"] += 1

    print(f"      [PartsList] Scanned page {page.number + 1}: {len(rows)} rows - confirmed "
          f"{stats['confirmed']}, part confirmed (qty not) {stats.get('part_confirmed', 0)}, "
          f"unconfirmed {stats['unconfirmed']}, added from vision only "
          f"{stats['vision_only']}" + ("" if vision else " (vision read unavailable - nothing confirmed)"))
    pno = page.number + 1
    return PartsList(page=pno, title=_strip_contd(title), subtitle=subtitle, rows=rows,
                     source="ocr+vision", verified=False, pages=[pno])


# ── Figure side ──────────────────────────────────────────────────────────────

def figure_callouts(page: fitz.Page, rect: fitz.Rect) -> set:
    """Callout numbers printed inside a figure - from the text layer, or OCR on
    a scanned page."""
    found = set()
    for x0, y0, x1, y1, text, conf in page_words(page):
        if conf < OCR_MIN_CONF_CALLOUT:
            continue
        if fitz.Rect(x0, y0, x1, y1).intersects(rect) and re.fullmatch(r"\d{1,3}", text):
            n = int(text)
            if 0 < n <= MAX_REF:
                found.add(str(n))
    return found


def page_text(page: fitz.Page) -> str:
    """The page's text for matching titles - printed, or OCR on a scan."""
    if page_has_text_layer(page):
        return page.get_text()
    return " ".join(w[4] for w in ocr_words(page))


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(text).lower()).strip()


def match_parts_list(figure_page_text: str, callouts: set, candidates: list) -> Optional[tuple]:
    """Choose the parts list that belongs to a figure.

    Evidence, strongest first: the list's subtitle or title is printed on the
    figure's page, and the list's refs cover the figure's callout numbers.
    Returns (parts_list, coverage, missing_callouts) or None.
    """
    page_norm = _norm_title(figure_page_text)
    best = None
    for pl in candidates:
        title_hit = any(
            t and _norm_title(t) and _norm_title(t) in page_norm
            for t in (pl.subtitle, pl.title)
        )
        coverage = (len(callouts & pl.refs) / len(callouts)) if callouts else 0.0
        # Title match is enough on its own only when the numbers don't
        # contradict it; numbers alone need to cover most of the drawing.
        if not ((title_hit and (not callouts or coverage >= 0.4)) or coverage >= 0.7):
            continue
        score = coverage + (1.0 if title_hit else 0.0)
        if best is None or score > best[0]:
            best = (score, pl, coverage)
    if not best:
        return None
    _, pl, coverage = best
    return pl, round(coverage, 2), sorted(callouts - pl.refs, key=int)


def merge_continuation(first: PartsList, following: list) -> PartsList:
    """Append rows from following pages that continue the same list."""
    key = (_norm_title(first.subtitle), _norm_title(first.title))
    for nxt in following:
        if (_norm_title(nxt.subtitle), _norm_title(nxt.title)) != key:
            break
        first.rows.extend(nxt.rows)
        first.pages.append(nxt.page)
        first.verified = first.verified and nxt.verified
    return first
