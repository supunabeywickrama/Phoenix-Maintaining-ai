"""
Parts lists printed as ruled tables on scanned or photographed pages - any
language, including Traditional Chinese lists (番號 / 品名 / 型式 / 數量 / 備註).

Measured on a real scanned hydraulic parts list (items 1-124, two tables side by
side, photographed at a slight angle):
- EasyOCR could not read the Chinese names at all and mangled the model codes
  ("KCG-3-250-D-Z-M-U-HL1-10" -> "K[6-3-250-0-7-H1-U-H[1-10");
- the vision model read codes correctly, but only from slices that contained
  whole columns - given a slice cut off mid-column it invented rows;
- each table was tilted ~1.9 deg, so rows drift into each other unless the
  table is straightened first.

So the table's own ruling lines drive everything: the table is straightened
from its rule lines, every row and column boundary is measured, vision reads
exact row slices (with the header on top so it knows the columns), and each
row's ref, code and quantity are read a second time by OCR from their own cell.
A row is confirmed only when both readings agree. Chinese names get an English
translation added for search, marked as a translation.
"""
import base64
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from unified_rag.ai_client import chat_json, MODEL_VISION, MODEL_CHAT_LIGHT
from services.llm_json import loads_tolerant
from services.parts_list import PartRow, PartsList, QTY_PATTERN, clean

SLICE_ROWS = 10
MIN_SLICE_ROWS = 3
MIN_ROW_RULES = 6
MIN_COL_RULES = 3
VISION_SCALE = 2.0            # slices are enlarged before the vision read

ROLE_WORDS = {
    "ref": ["番號", "番号", "項次", "项次", "序號", "序号", "編號", "编号", "no", "ref", "item", "件號", "件号"],
    "name": ["品名", "名稱", "名称", "零件名", "description", "name", "part name"],
    "model": ["型式", "型號", "型号", "規格", "规格", "part number", "part no", "model", "code", "p/n"],
    "qty": ["數量", "数量", "qty", "quantity", "q'ty"],
    "remarks": ["備註", "备注", "備考", "remarks", "remark", "note"],
}


@dataclass
class Table:
    image: np.ndarray        # straightened crop of the table
    rows: list               # y of each horizontal rule, top to bottom
    cols: list               # x of each vertical rule, left to right
    box: tuple               # (x, y, w, h) on the page image


# ── Grid ─────────────────────────────────────────────────────────────────────

def _masks(img: np.ndarray) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15)
    h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, img.shape[1] // 12), 1)))
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, img.shape[0] // 40))))
    return h, v


def _strokes(mask: np.ndarray, horizontal: bool, min_len: float) -> list:
    """Ruling strokes as pixel coordinate arrays. Components, not straight pixel
    rows: a tilted, slightly broken rule is still one stroke."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((3, 3), np.uint8)))
    out = []
    for i in range(1, n):
        if (st[i][2] if horizontal else st[i][3]) >= min_len:
            ys, xs = np.where(lab == i)
            out.append((xs, ys))
    return out


def find_tables(page_img: np.ndarray) -> list:
    """Ruled tables on the page, each straightened, with row and column rules.
    A drawing frame is also a rectangle of lines; it is not a table because it
    has almost no internal rules."""
    H, W = page_img.shape[:2]
    h, v = _masks(page_img)
    grid = cv2.dilate(cv2.bitwise_or(h, v), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables = []
    for c in contours:
        x, y, w, hh = cv2.boundingRect(c)
        if w * hh < 0.01 * W * H or w < 120 or hh < 120:
            continue
        # Generous margin: on a bent sheet the table's edge curves outside its
        # bounding box, and a tight crop cut the item numbers off the top rows.
        pad = max(12, int(0.04 * w))
        crop = page_img[max(0, y - pad):y + hh + pad, max(0, x - pad):x + w + pad]
        hm, _ = _masks(crop)
        slopes = [np.polyfit(xs, ys, 1)[0] for xs, ys in _strokes(hm, True, 0.5 * crop.shape[1]) if len(xs) > 50]
        if len(slopes) < MIN_ROW_RULES:
            continue
        angle = float(np.degrees(np.arctan(np.median(slopes))))
        ch, cw = crop.shape[:2]
        straight = cv2.warpAffine(crop, cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0), (cw, ch),
                                  borderValue=(255, 255, 255))
        # Levelling the rows is not enough: on a real scanned list the vertical
        # rules still leaned ~2 deg with the horizontal rules level (the sheet
        # was sheared, not just rotated). Over a 2100 px table that moved the
        # measured column lines ~37 px off the real ones at the top rows, so
        # every cell read landed in the wrong column. Shear the columns upright.
        _, vm = _masks(straight)
        leans = [np.polyfit(ys, xs, 1)[0] for xs, ys in _strokes(vm, False, 0.4 * ch) if len(ys) > 50]
        if leans:
            k = float(np.median(leans))
            if abs(k) > 0.002:
                shear = np.float32([[1, -k, k * ch / 2], [0, 1, 0]])
                straight = cv2.warpAffine(straight, shear, (cw, ch), borderValue=(255, 255, 255))
        hm2, vm2 = _masks(straight)
        rows = sorted(int(np.mean(ys)) for xs, ys in _strokes(hm2, True, 0.5 * cw))
        cols = sorted(int(np.mean(xs)) for xs, ys in _strokes(vm2, False, 0.4 * ch))
        # Merge double rules (a thick header underline scans as two lines): any
        # rule closer to the previous than 60% of a row is the same rule. Left
        # unmerged it added a phantom row, so 56 real items met "57 rows".
        gap = np.median(np.diff(rows)) if len(rows) > 2 else 0
        rows = [r for i, r in enumerate(rows) if i == 0 or r - rows[i - 1] > max(8, 0.6 * gap)]
        cols = [c2 for i, c2 in enumerate(cols) if i == 0 or c2 - cols[i - 1] > 8]
        if len(rows) < MIN_ROW_RULES or len(cols) < MIN_COL_RULES - 1:
            continue
        # A missing outer border (faint on the scan, or cut by the crop) is added
        # only when there is actual text beyond the outermost rule. Adding it on
        # extent alone created a phantom empty column on both real tables.
        body = (straight[rows[0]:rows[-1], :].min(axis=2) < 128)
        def has_text(x0, x1):
            return x1 - x0 > 6 and body[:, x0:x1].mean() > 0.01
        if cols[0] > 25 and has_text(0, cols[0] - 4):
            cols = [max(0, int(np.argmax(body.any(axis=0))) - 2)] + cols
        if cw - cols[-1] > 25 and has_text(cols[-1] + 4, cw):
            cols = cols + [min(cw - 1, int(cw - np.argmax(body.any(axis=0)[::-1])) + 2)]
        tables.append(Table(image=straight, rows=rows, cols=cols, box=(x, y, w, hh)))
    return sorted(tables, key=lambda t: (t.box[0], t.box[1]))


# ── Readers ──────────────────────────────────────────────────────────────────

def _vision(img: np.ndarray, prompt: str, max_tokens: int = 2500) -> Optional[dict]:
    big = cv2.resize(img, None, fx=VISION_SCALE, fy=VISION_SCALE, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", big)
    try:
        data = loads_tolerant(chat_json(MODEL_VISION, prompt, image_b64=base64.b64encode(buf.tobytes()).decode(),
                                        max_tokens=max_tokens, temperature=0.0) or "")
    except Exception as e:
        print(f"      ⚠️ [ScannedTable] Vision read failed: {e}")
        return None
    return data if isinstance(data, dict) else None


def _numbered_header(table: Table) -> np.ndarray:
    """The header row with a column number drawn above each column, so vision
    reports one header per column. Asked for a plain left-to-right list, it
    split spaced-out Chinese headers into single characters ("番 號" came back
    as two headers) and the count never matched the columns."""
    header = table.image[max(0, table.rows[0] - 2):table.rows[1] + 2, :].copy()
    strip = np.full((44, header.shape[1], 3), 255, np.uint8)
    for i in range(len(table.cols) - 1):
        cx = (table.cols[i] + table.cols[i + 1]) // 2
        cv2.putText(strip, str(i + 1), (max(0, cx - 9), 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.line(strip, (table.cols[i], 0), (table.cols[i], 43), (0, 0, 255), 1)
    return np.vstack([strip, header])


def _roles_from_headers(table: Table) -> dict:
    n_cols = len(table.cols) - 1
    data = _vision(_numbered_header(table), (
        f"Red numbers 1 to {n_cols} above this table header mark its columns. For each numbered "
        "column give the header text printed under that number, joining characters that are spaced "
        "apart (e.g. '番  號' is '番號'). It may be Chinese. "
        '{"columns": [{"n": 1, "header": "..."}]}'), max_tokens=300)
    roles, headers = {}, [""] * n_cols
    for item in (data or {}).get("columns") or []:
        try:
            n = int(item.get("n")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= n < n_cols:
            headers[n] = str(item.get("header", ""))
    for i, text in enumerate(headers):
        t = text.lower().replace(" ", "")
        for role, words in ROLE_WORDS.items():
            if role not in roles and t and any(w.replace(" ", "") in t for w in words):
                roles[role] = i
                break
    roles["_headers"] = headers
    return roles


def _roles_from_content(table: Table, roles: dict) -> dict:
    """Fill missing column roles from what the cells contain - independent of
    language or header wording. Item numbers count up; model codes are long
    letter-digit-hyphen strings; quantities are short numbers right of the code."""
    n_cols = len(table.cols) - 1
    n_rows = len(table.rows) - 2
    sample = list(range(1, n_rows + 1, max(1, n_rows // 8)))[:8]
    digits, codes = {}, {}
    for c in range(n_cols):
        if c in roles.values():
            continue
        reads_d = [_ocr_cell(_cell(table, r, c), "0123456789") for r in sample]
        reads_c = [_ocr_cell(_cell(table, r, c), "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-") for r in sample]
        nums = [int(x) for x in reads_d if x.isdigit()]
        digits[c] = (len(nums), nums)
        codes[c] = sum(1 for x in reads_c if len(x) >= 4 and re.search(r"[A-Z]", x) and re.search(r"\d", x))
    if "ref" not in roles:
        increasing = [c for c, (k, nums) in digits.items()
                      if k >= len(sample) * 0.6 and nums == sorted(nums) and len(set(nums)) > 1]
        if increasing:
            roles["ref"] = min(increasing)
    if "model" not in roles and codes:
        best = max(codes, key=codes.get)
        if codes[best] >= len(sample) * 0.5:
            roles["model"] = best
    if "qty" not in roles and "model" in roles:
        right = [c for c, (k, nums) in digits.items()
                 if c > roles["model"] and k >= len(sample) * 0.6 and nums and max(nums) < 1000]
        if right:
            roles["qty"] = min(right)
    if "name" not in roles and "ref" in roles and "model" in roles:
        between = [c for c in range(roles["ref"] + 1, roles["model"]) if c not in roles.values()]
        if between:
            roles["name"] = max(between, key=lambda c: table.cols[c + 1] - table.cols[c])
    return roles


def _column_roles(table: Table) -> Optional[dict]:
    """Which column is the ref, name, model code and quantity."""
    roles = _roles_from_headers(table)
    headers = roles.get("_headers", [])
    roles = _roles_from_content(table, roles)
    if "ref" not in roles or "model" not in roles:
        return None
    roles["_headers"] = [h or f"column {i + 1}" for i, h in enumerate(headers or [""] * (len(table.cols) - 1))]
    return roles


def _cell(table: Table, row: int, col: int) -> np.ndarray:
    y0, y1 = table.rows[row] + 2, table.rows[row + 1] - 1
    x0, x1 = table.cols[col] + 3, table.cols[col + 1] - 2
    return table.image[y0:y1, x0:x1]


def _ocr_cell(cell: np.ndarray, allowlist: str) -> str:
    from services.parts_list import _ocr_reader as reader_holder
    import services.parts_list as P
    if P._ocr_reader is None:
        import easyocr
        P._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    if cell.size == 0 or (cell.min(axis=2) < 128).mean() < 0.01:
        return ""
    gray = cv2.resize(cell.min(axis=2), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = np.pad(gray, 20, constant_values=255)
    found = P._ocr_reader.recognize(gray, allowlist=allowlist, detail=1)
    best = max(found, key=lambda f: f[2], default=None)
    return best[1].strip() if best and best[2] >= 0.3 else ""


def _fold(code: str) -> str:
    code = re.sub(r"[^A-Z0-9]", "", (code or "").upper())
    return code.replace("O", "0").replace("I", "1").replace("Q", "0")


def _read_slice(table: Table, first: int, last: int, roles: dict) -> Optional[list]:
    """Vision read of rows first..last (row indexes after the header). None when
    the reading does not have exactly one entry per ruled row."""
    header = table.image[max(0, table.rows[0] - 2):table.rows[1] + 2, :]
    body = table.image[table.rows[first] - 2:table.rows[last + 1] + 2, :]
    img = np.vstack([header, np.full((12, header.shape[1], 3), 255, np.uint8), body])
    n = last - first + 1
    headers = roles["_headers"]
    data = _vision(img, (
        f"This image is the header row of a parts list followed by exactly {n} table rows. Column headers "
        f"left to right: {headers}. Transcribe the {n} rows in order, one JSON object per row. Copy model / "
        "part codes character by character. A 〃 or \" mark means 'same as the row above' - write out the "
        "value it repeats. Use \"\" for an empty cell. "
        f'Return JSON {{"rows": [{{"no": str, "name": str, "model": str, "qty": str, "remarks": str}}]}} with exactly {n} rows.'))
    rows = (data or {}).get("rows") or []
    return rows if len(rows) == n else None


SLICE_STEP = 5          # half of SLICE_ROWS: every row falls inside two bands


def _slice_read(table: Table, top_rule: int, bottom_rule: int) -> list:
    """Vision read of one horizontal band of the table, header included.
    Returns the rows it could read in full. The band edges are padded by half a
    row: on a bent sheet a straight cut goes through text, so partly visible
    rows at the edges are expected and skipped - the overlapping band reads them."""
    row_h = max(12, int(np.median(np.diff(table.rows))))
    header = table.image[max(0, table.rows[0] - 4):table.rows[1] + 4, :]
    y0 = max(table.rows[1] + 2, table.rows[top_rule] - row_h // 2)
    y1 = min(table.image.shape[0], table.rows[bottom_rule] + row_h // 2)
    band = table.image[y0:y1, :]
    img = np.vstack([header, np.full((14, header.shape[1], 3), 255, np.uint8), band])
    data = _vision(img, (
        "The top line of this image is the header row of a machine parts list; below the gap is a band "
        "of its rows (the text may be Traditional Chinese). Transcribe every row whose ITEM NUMBER is fully "
        "visible, top to bottom. Skip a row cut off at the top or bottom edge. Copy the model / part code "
        "character by character. A 〃 or \" mark in a cell means 'same as the row above' - write out the "
        "value it repeats. Use \"\" for an empty cell. Do not invent rows. "
        '{"rows": [{"no": "item number", "name": "part name", "model": "model or part code", '
        '"qty": "quantity", "remarks": "remarks"}]}'))
    out = []
    for r in (data or {}).get("rows") or []:
        if not isinstance(r, dict):
            continue
        no = re.sub(r"\D", "", str(r.get("no", "")))
        if no:
            out.append({k: clean(str(r.get(k, ""))) for k in ("name", "model", "qty", "remarks")} | {"no": no})
    return out


def _agree(values: list, fold=lambda v: v) -> tuple:
    """(value, count agreeing) - the most common non-empty reading."""
    counts = {}
    for v in values:
        if v:
            key = fold(v)
            counts.setdefault(key, []).append(v)
    if not counts:
        return "", 0
    key = max(counts, key=lambda k: len(counts[k]))
    return counts[key][0], len(counts[key])


_LOOKALIKE = str.maketrans({"Z": "2", "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6"})


def _flag_lookalike_codes(rows: list) -> None:
    """Catch a misread both readings share. Reading the same pixels twice does
    not catch a mistake the model makes consistently: item 8 came back as
    "A3-606Z" from both bands while items 13, 21 and 26 read "A3-6062". Codes that
    differ only by look-alike characters (Z/2, O/0, I/1, S/5, B/8) are compared
    across the whole list; a rarer spelling is un-confirmed and the more common
    one is given as the likely reading. Nothing is changed automatically - two
    genuinely different parts can differ by one such character."""
    groups = {}
    for r in rows:
        if r.part_number:
            key = re.sub(r"[^A-Z0-9]", "", r.part_number.upper()).translate(_LOOKALIKE)
            groups.setdefault(key, []).append(r)
    for members in groups.values():
        spellings = {}
        for r in members:
            spellings.setdefault(r.part_number.upper(), []).append(r)
        if len(spellings) < 2:
            continue
        top = max(len(rs) for rs in spellings.values())
        leaders = [sp for sp, rs in spellings.items() if len(rs) == top]
        for sp, rs in spellings.items():
            # Only a spelling that is strictly the most common stays confirmed.
            # On a tie (2 x "A3-606Z", 2 x "A3-6062") the list cannot tell which
            # is right, so every row carrying either spelling is flagged.
            if sp in leaders and len(leaders) == 1:
                continue
            others = [o for o in spellings if o != sp]
            items = [x.ref for o in others for x in spellings[o]][:4]
            for r in rs:
                note = (f"code may be misread - look-alike spelling '{' / '.join(others)}' on items "
                        f"{', '.join(items)}; compare character by character")
                r.check = "UNCONFIRMED - " + note + " - check against the manual" if not r.check.startswith("UNCONFIRMED")                     else r.check.replace(" - check against the manual", f"; {note} - check against the manual")


def read_parts_table(table: Table) -> Optional[list]:
    """[PartRow] for a ruled parts-list table, or None if it is not one.

    Every row is read in two overlapping bands, so it is seen twice in
    different surroundings. The item number is the key that joins readings, so
    a band that drifts by a row cannot move a code onto the wrong item. A row
    is confirmed only when both readings agree on its code, and the item numbers
    run without gaps across the whole table.
    """
    n_rules = len(table.rows)
    n_rows = n_rules - 2
    readings = {}
    # Bands SLICE_ROWS tall every SLICE_STEP rows, plus a half-height band at
    # each end, so the first and last rows are also read twice. Measured before
    # this: with a 6-row step many rows (all of the first six) were read once.
    last = n_rules - 1
    plan = [(1, min(last, 1 + SLICE_STEP))]
    top = 1
    while True:
        plan.append((top, min(last, top + SLICE_ROWS)))
        if top + SLICE_ROWS >= last:
            break
        top += SLICE_STEP
    plan.append((max(1, last - SLICE_STEP), last))
    plan = list(dict.fromkeys(plan))
    for top, bottom in plan:
        for r in _slice_read(table, top, bottom):
            readings.setdefault(int(r["no"]), []).append(r)
    bands = len(plan)

    if len(readings) < MIN_SLICE_ROWS:
        return None
    codes_seen = sum(1 for rs in readings.values() if any(re.search(r"[A-Za-z0-9]{2,}", x["model"]) for x in rs))
    if codes_seen < len(readings) * 0.5:
        return None          # a ruled table, but not a parts list

    # The item-number range is the window of n_rows consecutive numbers that
    # holds the most readings. Taking min..max let one misread number ("10"
    # for 110) stretch a 69-124 table to 10-124 and fail every row.
    keys = sorted(readings)
    lo = max(keys, key=lambda k: sum(1 for x in keys if k <= x < k + n_rows))
    window = [x for x in keys if lo <= x < lo + n_rows]
    hi = max(window)
    stray = [x for x in keys if x not in window]
    expected = list(range(lo, hi + 1))
    gaps = [n for n in expected if n not in readings]
    # One ruled row of slack: a partial rule at the table edge is not a missing item.
    structure_ok = not gaps and abs(len(expected) - n_rows) <= 1
    if stray:
        print(f"      [ScannedTable] Ignored item numbers outside the table's run {lo}-{hi}: {stray[:8]}")

    rows, prev_name = [], ""
    for n in expected:
        rs = readings.get(n, [])
        model, model_votes = _agree([x["model"] for x in rs], _fold)
        name, _ = _agree([x["name"] for x in rs])
        qty, qty_votes = _agree([x["qty"] for x in rs if QTY_PATTERN.match(x["qty"] or "")])
        remarks, _ = _agree([x["remarks"] for x in rs])
        # A ditto mark means "same as above". Vision either writes the mark or
        # leaves the cell empty, which stored nameless parts ("part ref 105: ,
        # part number DSG-03-3C2"), so an empty name with a code repeats too.
        if name in ("〃", '"', "''", "同上", "〃〃", ",,", "//") or (not name and model and prev_name):
            name = prev_name
        prev_name = name or prev_name
        notes = []
        if not rs:
            notes.append("row not readable on the scan")
        elif len(rs) < 2:
            notes.append("read once only")
        if rs and model_votes < 2:
            variants = sorted({x["model"] for x in rs if x["model"]})
            notes.append("model code not confirmed" + (f" (readings: {', '.join(variants)})" if len(variants) > 1 else ""))
        if rs and qty_votes < 2:
            notes.append("quantity not confirmed")
        if not structure_ok:
            notes.append(f"item numbers {lo}-{hi} do not match the table's {n_rows} ruled rows"
                         + (f" (missing {gaps[:6]})" if gaps else ""))
        if rs and model_votes >= 2 and qty_votes >= 2 and structure_ok:
            check = "confirmed by two readings"
        elif rs and model_votes >= 2 and structure_ok:
            check = "part confirmed by two readings; " + "; ".join(notes)
        else:
            check = "UNCONFIRMED - " + "; ".join(notes) + " - check against the manual"
        rows.append(PartRow(ref=str(n), part_number=model, description=name, remarks=remarks,
                            qty=qty, check=check))
    _flag_lookalike_codes(rows)
    print(f"      [ScannedTable] {bands} bands read; items {lo}-{hi}; ruled rows {n_rows}; "
          f"structure {'OK' if structure_ok else 'MISMATCH'}")
    return rows


# ── Translation ──────────────────────────────────────────────────────────────

_CJK = re.compile(r"[㐀-鿿]")


def add_translations(rows: list) -> None:
    """Chinese part names get an English translation appended, marked as such,
    so a search in either language finds the part. Model codes are never
    touched. One call for all distinct names on the list."""
    names = sorted({r.description for r in rows if r.description and _CJK.search(r.description)})
    if not names:
        return
    prompt = ("Translate these Traditional Chinese machine-part names (hydraulic / injection moulding "
              "machine parts list) into short standard English technical terms. Return JSON "
              '{"translations": {"<chinese>": "<english>"}} covering every name. If a name looks misprinted '
              "or unclear, translate the most likely term and do not invent detail.\n" + "\n".join(names))
    try:
        data = loads_tolerant(chat_json(MODEL_CHAT_LIGHT, prompt, max_tokens=1500, temperature=0.0) or "")
        found = (data or {}).get("translations") or {}
    except Exception as e:
        print(f"      ⚠️ [ScannedTable] Translation failed: {e}")
        return
    for r in rows:
        en = str(found.get(r.description, "")).strip()
        if en and not _CJK.search(en):
            r.description = f"{r.description} (machine translation: {en})"


def read_scanned_parts_lists(page_img: np.ndarray, page_no: int, title: str = "") -> list:
    """Every ruled parts-list table on a prepared (upright) page image."""
    lists = []
    for table in find_tables(page_img):
        rows = read_parts_table(table)
        if not rows:
            continue
        add_translations(rows)
        confirmed = sum(1 for r in rows if r.check.startswith(("confirmed", "part confirmed")))
        print(f"      [ScannedTable] Page {page_no}: parts table with {len(rows)} rows "
              f"(refs {rows[0].ref}-{rows[-1].ref}); ref and code confirmed on {confirmed}")
        lists.append(PartsList(page=page_no, title=title, subtitle="", rows=rows,
                               source="grid+vision+ocr", verified=False, pages=[page_no]))
    return lists
