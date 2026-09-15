"""
One scanned or photographed page (already made upright by page_prep) turned
into the same kinds of chunks a digital page produces: a whole figure with its
callouts, parts-list tables, other tables, and page text.

What a page holds is decided by vision first, because these kinds need
different readers and a wrong reader invents content - asked to read a
controller screen as a parts list, a vision model produced item rows.
"""
import base64
import io
import re

import cv2
import numpy as np

from unified_rag.ai_client import chat_json, MODEL_VISION
from services.llm_json import loads_tolerant
from services import callouts as CO
from services import scanned_table as ST
from services.parts_list import PartsList, clean
from services.figure_validator import classify_figure

PAGE_TYPES = {"drawing", "parts_list", "parts_list_and_drawings", "table_and_drawing",
              "controller_screen", "text"}
OVERVIEW_LONG_SIDE = 1800
MIN_FIGURE_INK = 0.004


def _small(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _vision(img: np.ndarray, prompt: str, max_tokens: int = 1200) -> dict:
    ok, buf = cv2.imencode(".png", img)
    try:
        data = loads_tolerant(chat_json(MODEL_VISION, prompt, image_b64=base64.b64encode(buf.tobytes()).decode(),
                                        max_tokens=max_tokens, temperature=0.0) or "")
    except Exception as e:
        print(f"      ⚠️ [ScannedPage] Vision call failed: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def overview(img: np.ndarray) -> dict:
    """Page type, title and drawing numbers. Used for routing and pairing only -
    nothing from here is stored as a fact about a part."""
    data = _vision(_small(img, OVERVIEW_LONG_SIDE), (
        "This is one page from a machine manual (scanned or photographed; text may be Traditional "
        "Chinese or English). Return JSON:\n"
        '  "page_type": one of "drawing", "parts_list", "parts_list_and_drawings", "table_and_drawing", '
        '"controller_screen", "text".\n'
        "      parts_list = a table with item numbers, part names and model / part codes. "
        "controller_screen = a photo or print of a machine control panel screen. "
        "table_and_drawing = a drawing with a specification table that is not a numbered parts list.\n"
        '  "title": the sheet title exactly as printed in the title block (圖名) or the page heading. '
        'NOT a small sub-drawing label such as "A2-4422" or "A3-2590" - those go in drawing_numbers. '
        'Use "" if the title block cannot be read.\n'
        '  "machine_model": model text from the title block (型式), or "".\n'
        '  "drawing_numbers": every drawing number printed (e.g. "CA3-15583-2", "OTHERC16", "A3-2590"). '
        'Not fastener or thread specs like "4-M12x180L" or "PT1/4".\n'
        '  "callout_style": "circled_numbers", "item_codes" (labels like B08-13), "numbers" or "none".'))
    kind = str(data.get("page_type", "")).strip().lower()
    numbers = []
    for x in data.get("drawing_numbers") or []:
        n = clean(str(x)).strip("-").upper()
        # A drawing number has letters and digits and some length. Rejected:
        # bolt / thread specs ("4-M12x180L", "PT1/4"), screen page labels
        # ("PAGE-09"), and single characters (one page's "B-87-1" came back
        # split into "-", "B", "8", "7").
        if (len(n) >= 4 and re.search(r"[A-Z]", n) and re.search(r"\d", n)
                and not re.match(r"^(\d+-)?M\d|^PT\d|^PAGE-?\d", n)):
            numbers.append(n)
    numbers = list(dict.fromkeys(numbers))
    # The model sometimes loops: a hydraulic circuit page came back with 126
    # invented numbers counting A3-4419, A3-4420 ... A3-4544. A list that long,
    # or with a long run of consecutive numbers, is not trusted at all.
    tails = sorted(int(m.group(1)) for m in (re.search(r"(\d+)$", n) for n in numbers) if m)
    longest_run = run = 1
    for a, b in zip(tails, tails[1:]):
        run = run + 1 if b == a + 1 else 1
        longest_run = max(longest_run, run)
    if len(numbers) > 15 or longest_run >= 5:
        print(f"      ⚠️ [ScannedPage] Drawing numbers discarded as unreliable ({len(numbers)} returned)")
        numbers = []
    return {
        "page_type": kind if kind in PAGE_TYPES else "drawing",
        "title": clean(str(data.get("title", ""))),
        "machine_model": clean(str(data.get("machine_model", ""))),
        "drawing_numbers": numbers,
        "callout_style": str(data.get("callout_style", "")).strip().lower(),
    }


def transcribe_text(img: np.ndarray) -> str:
    """All readable text on a page or screen, kept as label: value lines."""
    data = _vision(_small(img, 2200), (
        "Transcribe all text visible on this machine manual page or controller screen photo. Keep each "
        "label with its value on one line (e.g. 'CLAMPING: 310', 'SAFETY TIME sec: 30.00 / 16.25'). Keep "
        "table layouts as rows of values separated by ' | '. Copy numbers exactly; write [unreadable] "
        'where you cannot read something - do not guess. Return JSON {"text": "..."}'), max_tokens=3000)
    return str(data.get("text", "")).strip()


def transcribe_table(table: "ST.Table") -> str:
    """A ruled table that is not a numbered parts list (e.g. a seal size chart),
    as a markdown table."""
    data = ST._vision(table.image, (
        "Transcribe this table exactly. Return JSON {\"header\": [column headings], \"rows\": [[cell, ...]]}. "
        "Copy codes character by character; use \"\" for empty cells and [unreadable] where you cannot read."),
        max_tokens=3000)
    header = [clean(str(h)) for h in data.get("header") or []] if data else []
    rows = [[clean(str(c)) for c in r] for r in (data or {}).get("rows") or [] if isinstance(r, list)]
    if not header or not rows:
        return ""
    width = len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        r = (r + [""] * width)[:width]
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def _figure_region(img: np.ndarray, exclude: list) -> tuple:
    """(crop, (x, y, w, h)) of the page's drawing content with tables blanked,
    or (None, None) when nothing but tables is there."""
    work = img.copy()
    for x, y, w, h in exclude:
        work[max(0, y - 6):y + h + 6, max(0, x - 6):x + w + 6] = 255
    ink = (cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) < 150).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # scan speckle
    if ink.mean() < MIN_FIGURE_INK:
        return None, None
    ys, xs = np.where(ink > 0)
    H, W = ink.shape
    x0, x1 = max(0, int(np.percentile(xs, 0.2)) - 20), min(W, int(np.percentile(xs, 99.8)) + 20)
    y0, y1 = max(0, int(np.percentile(ys, 0.2)) - 20), min(H, int(np.percentile(ys, 99.8)) + 20)
    return work[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def _callout_sort(c: str):
    return (0, int(c)) if c.isdigit() else (1, c)


def parse_scanned_page(page, page_idx: int, manual_id: str, cloudinary) -> dict:
    """{"chunks": [...], "figure": figures-entry or None, "parts_list": PartsList or None}"""
    img = CO.page_image(page)
    info = overview(img)
    kind = info["page_type"]
    title = info["title"] or f"Page {page_idx}"
    heading = " ".join(filter(None, [info["title"], info["machine_model"]])) or f"Page {page_idx}"
    print(f"      [ScannedPage] Page {page_idx}: {kind} | title '{info['title']}' | model "
          f"'{info['machine_model']}' | drawings {info['drawing_numbers']}")

    chunks, parts_list, exclude = [], None, []
    page_text = "\n".join(filter(None, [info["title"], info["machine_model"], " ".join(info["drawing_numbers"])]))

    # Tables
    tables = ST.find_tables(img) if kind in ("parts_list", "parts_list_and_drawings", "table_and_drawing") else []
    part_rows = []
    page_area = img.shape[0] * img.shape[1]
    for table in tables:
        x, y, w, h = table.box
        if w * h > 0.35 * page_area:
            # A drawing frame with its title block has ruling lines too. Read as a
            # "table" and blanked out, it erased a whole exploded-view drawing
            # (JW-600SP clamp system) - a real table never fills most of a page.
            continue
        rows = ST.read_parts_table(table) if kind != "table_and_drawing" else None
        if rows:
            part_rows.extend(rows)
            exclude.append(table.box)
            continue
        md = transcribe_table(table) if kind == "table_and_drawing" else ""
        if md:
            exclude.append(table.box)
            page_text += "\n" + md[:1500]
            chunks.append({"type": "table", "page": page_idx, "kind": "table",
                           "content": f"{heading} — table\n{md}", "render_markdown": md,
                           "metadata": {"title": f"{heading} — table", "parts_list": False,
                                        "verified": False, "section": heading}})
    if part_rows:
        part_rows.sort(key=lambda r: _callout_sort(r.ref) if r.ref else (2, ""))
        ST.add_translations(part_rows)
        list_title = info["title"]
        parts_list = PartsList(page=page_idx, title=list_title, subtitle=info["machine_model"], rows=part_rows,
                               source="grid+vision", verified=False, pages=[page_idx])
        parts_list.drawing_numbers = info["drawing_numbers"]
        chunks.append({"type": "table", "page": page_idx, "kind": "table",
                       "content": parts_list.to_text(), "render_markdown": parts_list.to_markdown(),
                       "metadata": {"title": f"{heading} — parts list", "caption": "", "parts_list": True,
                                    "verified": False, "section": heading}})

    # Screens and text pages: the words are the content.
    # A photographed controller screen was typed "table_and_drawing" on two of
    # three pages, which skipped its text - so those pages transcribe too.
    if kind in ("controller_screen", "text", "table_and_drawing"):
        text = transcribe_text(img)
        if text:
            chunks.append({"type": "text", "page": page_idx,
                           "content": f"{heading} (page {page_idx}, read from a {kind.replace('_', ' ')} "
                                      f"photo/scan)\n{text}",
                           "metadata": {"section": heading}})
            page_text += "\n" + text[:1500]

    # The drawing - whole, never split: scanned drawing sheets hold many views
    # joined by dashed lines and a shared callout numbering.
    figure = None
    crop, box = _figure_region(img, exclude)
    if crop is not None and kind != "parts_list":
        verdict = classify_figure(_small(crop, 1400), page_text)
        if verdict["keep"]:
            nums = CO.circled_numbers(img, exclude=exclude)
            callouts = {str(n) for n, *_ in nums}
            if len(callouts) < 3 and kind in ("drawing", "parts_list_and_drawings", "table_and_drawing"):
                callouts |= {c for c, *_ in CO.item_codes(img, exclude=exclude)}
            h, w = crop.shape[:2]
            ok, buf = cv2.imencode(".png", _small(crop, 4000))
            url = cloudinary.upload_image(io.BytesIO(buf.tobytes()), f"{manual_id}_p{page_idx}_fig0")
            if url:
                chunk = {
                    "type": "image", "path": url, "page": page_idx,
                    "figure_role": "full", "parent_path": None, "width": w, "height": h,
                    "kind": "photo" if kind == "controller_screen" else verdict["kind"],
                    "metadata": {
                        "section": heading, "label": "Full Diagram", "figure_role": "full",
                        "page_text": page_text, "layout": verdict["layout"],
                        "callouts": sorted(callouts, key=_callout_sort),
                        "drawing_code": (info["drawing_numbers"] or [""])[0],
                        "drawing_numbers": info["drawing_numbers"],
                        "item_codes": sorted(c for c in callouts if not c.isdigit()),
                        "scanned": True,
                    },
                }
                chunks.append(chunk)
                figure = {"chunk": chunk, "page": page_idx, "callouts": callouts,
                          "page_text": page_text, "layout": verdict["layout"],
                          "drawing_numbers": info["drawing_numbers"]}
                print(f"      [ScannedPage] Figure kept whole ({verdict['kind']}); callouts read: "
                      f"{len(callouts)} {sorted(callouts, key=_callout_sort)[:12]}")
    return {"chunks": chunks, "figure": figure, "parts_list": parts_list}
