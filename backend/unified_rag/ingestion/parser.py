import copy
import fitz  # PyMuPDF
import io
import logging
import re
import numpy as np
import cv2
from PIL import Image
from services.cloudinary_service import CloudinaryService

# Suppress pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    import camelot
except ImportError:
    camelot = None

def _df_to_markdown(df) -> str:
    """Render a camelot DataFrame as a markdown table.

    Kept alongside the LLM summary rather than instead of it: the summary is what
    gets embedded and searched, this is what the technician actually reads.
    """
    try:
        rows = df.fillna("").astype(str).values.tolist()
        if not rows:
            return ""
        header = [c.strip().replace("\n", " ") or f"Col {i+1}" for i, c in enumerate(rows[0])]
        body = rows[1:]
        out = ["| " + " | ".join(header) + " |",
               "| " + " | ".join("---" for _ in header) + " |"]
        for r in body:
            cells = [str(c).strip().replace("\n", " ").replace("|", "\\|") for c in r]
            cells += [""] * (len(header) - len(cells))
            out.append("| " + " | ".join(cells[:len(header)]) + " |")
        return "\n".join(out)
    except Exception as e:
        print(f"      [Parser] Could not render table as markdown: {e}")
        return ""


_CAPTION_RE = re.compile(
    # "table" alone would also match a "Table of Contents" heading, hence the
    # required digit after it; the others are specific enough on their own.
    r"^(key to fig(?:ure)?\.?\s*\d|legend\b|parts?\s*list\b|table\s*\d)", re.IGNORECASE
)


def _guess_table_caption(page_text: str) -> str:
    """First line on the page that reads like a table/legend heading, e.g.
    'Key to Figure 1.1'. Camelot only returns the grid, never the caption
    printed above it, so without this every table's title defaults to
    'Table on page N' and a parts-legend can't be told apart from any other
    grid on the same page."""
    for line in (page_text or "").splitlines():
        line = line.strip()
        if line and _CAPTION_RE.match(line):
            return line[:100]
    return ""


def _looks_like_legend(df) -> bool:
    """True when the first data column is a run of small integers - the
    'Item 1, Item 2, ...' shape a figure key has.

    A page often prints a "Key to Figure 1.1" heading above the legend AND a
    second, unrelated table (engine specs, torque figures) below it. Without
    this check the heading gets stamped on both, and the spec table shows up
    in chat titled "Key to Figure 1.1" - confirmed in real ingested data.
    """
    try:
        rows = df.fillna("").astype(str).values.tolist()[1:]
        if len(rows) < 3:
            return False
        ints = sum(
            1 for r in rows
            if r and r[0].strip().isdigit() and 0 < int(r[0].strip()) < 200
        )
        return ints >= max(3, len(rows) // 2)
    except Exception:
        return False


def _header_title(df) -> str:
    """A title built from the table's own header row. Deterministic on
    purpose: an LLM asked to title a table invents manual-style headings
    ("Figure 1.7: ...") that the manual never printed."""
    try:
        header = [c.strip().replace("\n", " ")
                  for c in df.fillna("").astype(str).values.tolist()[0]]
        seen, uniq = set(), []
        for h in header:
            if h and not h.isdigit() and h.lower() not in seen:
                seen.add(h.lower())
                uniq.append(h)
        return " / ".join(uniq[:4])[:80] if len(uniq) >= 2 else ""
    except Exception:
        return ""


def _table_title(df, caption: str, section: str) -> str:
    """Caption if it genuinely belongs to this table, else the table's own
    header row, else the section we're in (the old behaviour, kept last
    because `current_section` leaks across everything on the page)."""
    if caption:
        is_legend_caption = bool(
            re.match(r"^(key to fig|legend|parts?\s*list)", caption, re.IGNORECASE)
        )
        if not is_legend_caption or _looks_like_legend(df):
            return caption
    return _header_title(df) or section


def _df_to_text(df, caption: str = "") -> str:
    """Flatten a table into readable 'Header: value' lines instead of a raw
    JSON dump. This becomes the chunk's embedded/searched content (and its
    displayed title, via the first line) - a legend table only helps someone
    ask "what is item 3" if the numbers and names it pairs are actually
    readable text, not `{"0":{"0":"1",...}}`.
    """
    try:
        rows = df.fillna("").astype(str).values.tolist()
        if not rows:
            return caption
        header = [c.strip().replace("\n", " ") or f"Col {i+1}" for i, c in enumerate(rows[0])]
        lines = [caption] if caption else []
        for r in rows[1:]:
            pairs = [
                f"{h}: {v.strip()}" for h, v in zip(header, r)
                if v and v.strip() and not h.lower().startswith("col ")
            ]
            if not pairs:
                # Header row itself is unlabeled (common for these grids) -
                # fall back to raw cell values so nothing is silently dropped.
                pairs = [v.strip() for v in r if v and v.strip()]
            if pairs:
                lines.append(" | ".join(pairs))
        text = "\n".join(lines).strip()
        return text or caption
    except Exception as e:
        print(f"      [Parser] Could not flatten table to text: {e}")
        return caption


from services.table_validator import is_valid_table

# A "composite" figure split into more pieces than this was really one assembly
# drawing (the Cab Windows exploded view came back as ~50 pieces) - kept whole.
MAX_SPLIT_VIEWS = 6

_DRAWING_CODE = re.compile(r"^[A-Z]{1,3}-{1,2}\d{2,6}$")


def _expand_to_frame(page_bgr, box, pad_ratio=0.015):
    """Grow a layout-model figure box out to the rectangular frame drawn around
    the figure, if there is one; otherwise add a small margin.

    Not every figure has a frame, so both cases are handled. This only moves
    the crop's edges outward - it never drops anything. Verified on the 873
    parts manual: the raw layout box cut off callout 28 on Cab Windows (p.226),
    callout 30 on the Caliper Brake Kit (p.246) and the drawing codes on both;
    the frame-grown crop contains all of them. On an unframed figure (TPM-750
    p.14) the margin recovered a clipped "1 000 mm" dimension label.
    """
    H, W = page_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < 160).astype(np.uint8) * 255
    # Long straight strokes only: a frame is made of lines at least half the
    # figure's width/height, which leader lines and part outlines are not.
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(bw * 0.5)), 1)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(bh * 0.5)))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in contours:
        fx, fy, fw, fh = cv2.boundingRect(c)
        ix = max(0, min(x2, fx + fw) - max(x1, fx))
        iy = max(0, min(y2, fy + fh) - max(y1, fy))
        covers = (ix * iy) / float(bw * bh)
        size_ratio = (fw * fh) / float(bw * bh)
        # The frame must hold nearly all of the detected figure and be about
        # its size - not the page border or a table ruling elsewhere.
        if covers >= 0.9 and 0.8 <= size_ratio <= 1.6:
            if best is None or fw * fh < best[2] * best[3]:
                best = (fx, fy, fw, fh)
    if best:
        fx, fy, fw, fh = best
        m = 4
        return (max(0, fx - m), max(0, fy - m), min(W, fx + fw + m), min(H, fy + fh + m)), "frame"
    px, py = int(W * pad_ratio), int(H * pad_ratio)
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)), "no frame, padded"


FURNITURE_BAND = 0.08        # top/bottom share of the page where headers/footers sit
FURNITURE_MIN_PAGES = 3
FURNITURE_MIN_SHARE = 0.30   # ...and on at least this share of all pages


def _furniture_key(text: str) -> str:
    """Page numbers and other digits masked, so "225 Model 873" == "226 Model 873"."""
    return re.sub(r"\d+", "#", " ".join((text or "").lower().replace("--", "-").split())).strip()


def _page_furniture(doc) -> set:
    """Running header/footer lines, found from the printed text of EVERY page.

    Counting repeated text chunks instead (the first version) needed the
    footer to become a chunk on 3+ pages; on a test run only 2 pages produced
    one, so "Model 873 G--Series" stayed in the index. Here each page's own
    text in the top and bottom bands is looked at directly, so a footer is
    recognised however few chunks it happened to end up in.
    """
    counts = {}
    for page in doc:
        height = page.rect.height
        seen = set()
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y1 <= height * FURNITURE_BAND or y0 >= height * (1 - FURNITURE_BAND):
                    key = _furniture_key(" ".join(s["text"] for s in line["spans"]))
                    if key and key != "#":
                        seen.add(key)
        for key in seen:
            counts[key] = counts.get(key, 0) + 1
    need = max(FURNITURE_MIN_PAGES, int(len(doc) * FURNITURE_MIN_SHARE))
    return {k for k, c in counts.items() if c >= need}


def _drop_page_furniture(parsed_data: list, doc) -> list:
    """Remove text chunks made up entirely of running headers/footers and page
    numbers. A chunk with any real content on top of the footer is kept."""
    if len(doc) < FURNITURE_MIN_PAGES:
        return parsed_data
    furniture = _page_furniture(doc)
    if not furniture:
        return parsed_data

    def only_furniture(content: str) -> bool:
        pieces = [p for p in re.split(r"\n| \| ", (content or "").replace("Page contents:", "")) if p.strip()]
        return bool(pieces) and all(
            _furniture_key(p) in furniture or re.fullmatch(r"#+", _furniture_key(p)) for p in pieces
        )

    kept = [item for item in parsed_data
            if not (item.get("type") == "text" and only_furniture(item.get("content")))]
    dropped = len(parsed_data) - len(kept)
    if dropped:
        print(f"      [Parser] Dropped {dropped} header/footer-only text chunk(s); "
              f"furniture lines: {sorted(furniture)[:4]}")
    return kept


VIEW_GAP_RATIO = 0.03       # blank band, as a share of the figure, that separates views
VIEW_CAPTION_RATIO = 0.12   # a band shorter than this is a caption, joined to its view


def _split_views(crop) -> list:
    """Split a figure made of separate views along the blank gutters between them.

    Replaces point-prompted SAM segmentation for this job. Tested on TPM-750
    Figure 1.1 - views (a) and (b) stacked in one picture - SAM returned view (a)
    as a 23x17 px fragment and view (b) as a 372x64 strip. Separate views on a
    manual page are laid out with white space between them, which is a far more
    reliable boundary for line drawings than a segmentation mask.

    A short band is a caption ("(a) TPM-750-2M two-operator pruner") and is
    joined to the view above it rather than becoming a "view" of its own.
    Returns [{"crop", "label", "box"}] - empty when no clean split exists.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    ink = gray < 200
    for axis in (0, 1):            # 0: views stacked top-to-bottom, 1: side by side
        profile = ink.any(axis=1 - axis)
        size = len(profile)
        min_gap = max(8, int(size * VIEW_GAP_RATIO))
        bands, start, gap = [], None, 0
        for i, has_ink in enumerate(profile):
            if has_ink:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= min_gap:
                    bands.append([start, i - gap + 1])
                    start, gap = None, 0
        if start is not None:
            bands.append([start, size])

        merged = []
        for band in bands:
            if merged and (band[1] - band[0]) < size * VIEW_CAPTION_RATIO:
                merged[-1][1] = band[1]           # caption under the view above
            else:
                merged.append(band)
        if merged and len(merged) > 1 and (merged[0][1] - merged[0][0]) < size * VIEW_CAPTION_RATIO:
            merged[1][0] = merged[0][0]           # a caption above the first view
            merged.pop(0)
        if len(merged) < 2:
            continue

        views = []
        for n, (a, b) in enumerate(merged, 1):
            a, b = max(0, a - 6), min(size, b + 6)
            part = crop[a:b, :] if axis == 0 else crop[:, a:b]
            box = (0, a, crop.shape[1], b - a) if axis == 0 else (a, 0, b - a, crop.shape[0])
            views.append({"crop": part, "label": f"View {n} of {len(merged)}", "box": box})
        return views
    return []


def _overlap(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (ix * iy) / float(smaller) if smaller > 0 else 0.0


def _drawing_code(page, rect) -> str:
    """The drawing reference printed in the figure's corner, e.g. "TS-2007"."""
    from services.parts_list import page_words
    for x0, y0, x1, y1, text, *_ in page_words(page):
        if fitz.Rect(x0, y0, x1, y1).intersects(rect) and _DRAWING_CODE.match(text):
            return text.replace("--", "-")
    return ""


class DocumentParser:
    def __init__(self, yolo_weights="models/yolov8_doclaynet.pt"):
        self.cloudinary = CloudinaryService()
        
        # Step 1: Initialize Layout Detection (YOLOv8 DocLayNet)
        if YOLO:
            print("Initializing YOLOv8 DocLayNet Layout Detection...")
            try:
                self.layout_model = YOLO(yolo_weights)
                print(f"Successfully loaded YOLOv8 weights from {yolo_weights}")
            except Exception as e:
                print(f"WARNING: Could not load YOLOv8 model from {yolo_weights}. Error: {e}")
                self.layout_model = None
        else:
            print("WARNING: ultralytics (YOLO) not installed. Skipping AI layout detection.")
            self.layout_model = None

        # Initialize EasyOCR Reader
        if easyocr:
            print("Initializing EasyOCR...")
            self.reader = easyocr.Reader(['en'], gpu=False)
        else:
            print("WARNING: easyocr not installed. Skipping OCR fallback.")
            self.reader = None
        
    def extract_text_with_ocr(self, image_path):
        """Uses EasyOCR to extract text from a specific image/region."""
        if not self.reader:
            return ""
        results = self.reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        return text

    def parse_pdf(self, file_path: str, manual_id: str):
        """
        Processes PDF with Structural Context & Agentic Figure Splitting:
        - Maintains 'current_section' context for every item.
        - Uses YOLOv8 for layout Detection.
        - Splits a figure into its separate views only along blank gutters (_split_views).
        """
        from services.figure_validator import classify_figure
        from services.parts_list import (
            figure_callouts, match_parts_list, merge_continuation,
            page_has_text_layer, read_text_layer, read_scanned,
            page_text as scanned_page_text,
        )
        from services.page_prep import is_scanned
        from services.scanned_page import parse_scanned_page

        print(f"📄 [Parser] Opening PDF: {file_path}")
        doc = fitz.open(file_path)
        parsed_data = []
        total_pages = len(doc)
        figures = []        # every full figure kept, for pairing with parts lists
        parts_lists = {}    # page -> PartsList read from that page
        
        # In-memory tracking of the document's structural hierarchy
        current_section = "General Information"
        
        print(f"📖 [Parser] PDF has {total_pages} pages. Starting extraction...")
        
        for page_num in range(total_pages):
            page_idx = page_num + 1
            print(f"   ∟ Processing Page {page_idx}/{total_pages}...")
            page = doc.load_page(page_num)

            # A scanned or photographed page (made upright by services/page_prep
            # before parsing) has no text layer: layout detection sees it as one
            # big picture, and the text/table readers below find nothing. It
            # gets its own reader - see services/scanned_page.py.
            if is_scanned(page):
                try:
                    result = parse_scanned_page(page, page_idx, manual_id, self.cloudinary)
                except Exception as e:
                    print(f"      ⚠️ [Parser] Scanned page {page_idx} could not be read: {e}")
                    result = {"chunks": [], "figure": None, "parts_list": None}
                parsed_data.extend(result["chunks"])
                if result["figure"]:
                    figures.append(result["figure"])
                if result["parts_list"]:
                    parts_lists[page_idx] = result["parts_list"]
                continue

            # Full page text, independent of the YOLO-detected text blocks used
            # for chunking below. Captions are grounded against this so a vision
            # model describing a diagram cannot invent a different machine or
            # power source than what the manual itself states on the same page
            # (caught this for real: a diagram was captioned "pneumatic
            # reciprocating saw" when the manual's own text on that page reads
            # "TPM-750 Petrol ... two-stroke ... Pruning Machine").
            page_text = page.get_text("text").strip()

            # Step 1: Layout Detection
            pix = page.get_pixmap(dpi=150)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            if self.layout_model:
                results = self.layout_model(Image.fromarray(img_array), verbose=False)
                boxes = results[0].boxes
                names = self.layout_model.names

                img_index = 0
                # Per-page state for the two fixes below - reset every page.
                # 1) seen_text_on_page: the layout model's per-class NMS does not
                #    suppress overlapping boxes of DIFFERENT classes, so the same
                #    paragraph sometimes gets detected as both "Text" and
                #    "List-item" and extracted twice, verbatim. Confirmed in real
                #    data: ~4% of text chunks across a real manual were exact
                #    (page, content) duplicates.
                # 2) short_fragments: a table-of-contents or index page is a
                #    cluster of many small "Text"/"List-item" boxes, one per
                #    heading, each with no body content of its own. Storing each
                #    as its own chunk let a bare 4-word TOC entry like "Test 4 -
                #    Testing the Clutch" out-rank the real 150-word procedure on
                #    its own page in vector search, since the fragment is a
                #    near-exact phrase match with none of the surrounding noise a
                #    real answer has. Buffered here and merged into one chunk.
                seen_text_on_page = set()
                short_fragments = []
                figure_boxes_on_page = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = names[cls_id].lower()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    x_scale = page.rect.width / pix.width
                    y_scale = page.rect.height / pix.height
                    rect = fitz.Rect(x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale)
                    
                    # Update Section Header Context
                    if "title" in class_name or "header" in class_name:
                        txt = page.get_text("text", clip=rect).strip()
                        if txt and len(txt) > 3:
                            current_section = txt
                            print(f"      [Structure] New Section Detected: {current_section}")

                    # FIGURES
                    if "picture" in class_name or "figure" in class_name:
                        # The layout box routinely clips the edge of a framed
                        # figure - on the 873 parts manual it cut off callout 28
                        # (Cab Windows) and callout 30 (Caliper Brake Kit). Grow it
                        # to the frame, or pad it when there is no frame.
                        (ex1, ey1, ex2, ey2), grown = _expand_to_frame(
                            img_bgr, (int(x1), int(y1), int(x2), int(y2))
                        )
                        # Several layout boxes inside one frame are one figure.
                        if any(_overlap((ex1, ey1, ex2, ey2), seen) > 0.8 for seen in figure_boxes_on_page):
                            continue
                        figure_boxes_on_page.append((ex1, ey1, ex2, ey2))
                        rect = fitz.Rect(ex1 * x_scale, ey1 * y_scale, ex2 * x_scale, ey2 * y_scale)
                        raw_crop = img_bgr[ey1:ey2, ex1:ex2]

                        # Vision alone decides whether this is a figure, and
                        # whether it may be split. No pixel-size rule drops anything.
                        if not page_text and not page_has_text_layer(page):
                            # Scanned page: OCR it once so the figure's title can be
                            # matched to its parts list and grounds the caption.
                            page_text = scanned_page_text(page)
                        verdict = classify_figure(raw_crop, page_text)
                        if not verdict["keep"]:
                            print(f"      🚫 [Validator] Not a figure on page {page_idx}: "
                                  f"{verdict['kind']} - {verdict['reason']}")
                            continue

                        parent_ctx = f"Figure on Page {page_idx} under section '{current_section}'"
                        callouts = figure_callouts(page, rect)
                        drawing_code = _drawing_code(page, rect)
                        print(f"      [Figure] {verdict['kind']}/{verdict['layout']} ({grown}) - "
                              f"{len(callouts)} callout numbers in the text layer - {verdict['reason']}")

                        # Only separate views drawn side by side are split. An
                        # exploded parts view is kept whole: cutting it apart
                        # separates each callout number from the part it points
                        # at, which is the whole point of the drawing.
                        sub_figures = []
                        if verdict["layout"] == "composite_views":
                            try:
                                sub_figures = _split_views(raw_crop)
                            except Exception as e:
                                print(f"      ⚠️ [Parser] View split failed: {e}. Keeping whole.")
                                sub_figures = []
                            if not sub_figures:
                                print("      [Figure] No clean gap between views - kept whole.")
                            if len(sub_figures) > MAX_SPLIT_VIEWS:
                                # Dozens of "views" means it was really one
                                # assembly drawing; keep it whole.
                                print(f"      [Figure] Split produced {len(sub_figures)} pieces - "
                                      f"that is an assembly, not separate views. Keeping whole.")
                                sub_figures = []
                        else:
                            print(f"      [Figure] Kept whole ({verdict['layout']}).")

                        # The whole figure is always stored, even when it splits
                        # cleanly: an explanation should show the full drawing for
                        # orientation before zooming into one component.
                        #
                        # Re-rendered at higher DPI specifically for this upload:
                        # 150 DPI (used for layout detection) is fine for finding
                        # where a figure is, but a dense composite drawing with a
                        # dozen small circled callout numbers needs more than
                        # that to stay legible to the vision model captioning it.
                        # SAM splitting below still uses the original low-res
                        # raw_crop - its geometry doesn't need the extra detail.
                        try:
                            hi_pix = page.get_pixmap(clip=rect, dpi=300)
                            hi_arr = np.frombuffer(hi_pix.samples, dtype=np.uint8).reshape(
                                hi_pix.height, hi_pix.width, 3
                            )
                            full_crop = cv2.cvtColor(hi_arr, cv2.COLOR_RGB2BGR)
                        except Exception:
                            full_crop = raw_crop

                        fh, fw = full_crop.shape[:2]
                        _, buf = cv2.imencode(".png", full_crop)
                        full_url = self.cloudinary.upload_image(
                            io.BytesIO(buf.tobytes()), f"{manual_id}_p{page_idx}_fig{img_index}"
                        )

                        if full_url:
                            figure_chunk = {
                                "type": "image", "path": full_url, "page": page_idx,
                                "figure_role": "full", "parent_path": None,
                                "width": fw, "height": fh,
                                "kind": verdict["kind"],
                                "metadata": {
                                    "section": current_section,
                                    "label": "Full Diagram",
                                    "figure_role": "full",
                                    "page_text": page_text,
                                    "layout": verdict["layout"],
                                    "callouts": sorted(callouts, key=int),
                                    "drawing_code": drawing_code,
                                },
                            }
                            parsed_data.append(figure_chunk)
                            figures.append({"chunk": figure_chunk, "page": page_idx,
                                            "callouts": callouts, "page_text": page_text,
                                            "layout": verdict["layout"]})
                        else:
                            print(f"      [Parser] Upload unavailable - skipping figure (page {page_idx}).")

                        for i, sub in enumerate(sub_figures):
                            # Cut the view from the 300 dpi render, not the 150 dpi
                            # layout image the split was measured on.
                            bx, by, bw, bh = sub["box"]
                            sx = full_crop.shape[1] / max(raw_crop.shape[1], 1)
                            sy = full_crop.shape[0] / max(raw_crop.shape[0], 1)
                            hi = full_crop[int(by * sy):int((by + bh) * sy), int(bx * sx):int((bx + bw) * sx)]
                            if hi.size:
                                sub["crop"] = hi
                            if sub["crop"] is None or sub["crop"].size == 0:
                                continue
                            sub_verdict = classify_figure(sub["crop"], page_text)
                            if not sub_verdict["keep"]:
                                print(f"         [Validator] Dropped view: {sub_verdict['kind']} - "
                                      f"{sub_verdict['reason']}")
                                continue

                            sh, sw = sub["crop"].shape[:2]
                            _, buf = cv2.imencode(".png", sub["crop"])
                            cloud_url = self.cloudinary.upload_image(
                                io.BytesIO(buf.tobytes()), f"{manual_id}_p{page_idx}_sub{img_index}_{i}"
                            )

                            if cloud_url:
                                parsed_data.append({
                                    "type": "image", "path": cloud_url, "page": page_idx,
                                    "figure_role": "part", "parent_path": full_url,
                                    "width": sw, "height": sh,
                                    "metadata": {
                                        "section": current_section,
                                        "label": sub["label"],
                                        "parent_context": parent_ctx,
                                        "figure_role": "part",
                                        "page_text": page_text,
                                    },
                                })
                                print(f"         Isolated component: {sub['label']} ({sw}x{sh})")
                            else:
                                print(f"      [Parser] Upload unavailable - skipping sub-figure.")

                        img_index += 1
                        
                    # TEXT
                    elif "text" in class_name or "list" in class_name:
                        text_content = page.get_text("text", clip=rect).strip()
                        if not text_content:
                            continue

                        dedup_key = text_content.lower()
                        if dedup_key in seen_text_on_page:
                            continue
                        seen_text_on_page.add(dedup_key)

                        words = text_content.split()
                        looks_like_toc_entry = (
                            len(text_content) < 60 and len(words) <= 6
                            and not text_content.rstrip().endswith((".", "!", "?", ":"))
                        )
                        if looks_like_toc_entry:
                            short_fragments.append(text_content)
                        else:
                            parsed_data.append({
                                "type": "text", "content": text_content, "page": page_idx,
                                "metadata": {"section": current_section}
                            })

                # Flush buffered short fragments. 3+ on one page is the TOC/index
                # signature; fewer than that could be a genuine short standalone
                # note, so those are kept as individual chunks as before.
                if len(short_fragments) >= 3:
                    parsed_data.append({
                        "type": "text",
                        "content": "Page contents: " + " | ".join(short_fragments),
                        "page": page_idx,
                        "metadata": {"section": current_section},
                    })
                else:
                    for frag in short_fragments:
                        parsed_data.append({
                            "type": "text", "content": frag, "page": page_idx,
                            "metadata": {"section": current_section},
                        })
            else:
                # Basic Fallback logic
                blocks = page.get_text("blocks")
                for b in blocks:
                    if b[6] == 0:
                        txt = b[4].strip()
                        if txt: parsed_data.append({"type": "text", "content": txt, "page": page_idx, "metadata": {"section": current_section}})

            # Step 2a: Parts lists, read from the printed text (see
            # services/parts_list.py for why not camelot or vision).
            parts = read_text_layer(page) if page_has_text_layer(page) else None
            if parts:
                parts_lists[page_idx] = parts
                print(f"      [PartsList] {parts.title} {parts.subtitle}: {len(parts.rows)} rows, "
                      f"{len(parts.refs)} refs (from printed text)")
                parsed_data.append({
                    "type": "table", "page": page_idx,
                    "content": parts.to_text(),
                    "kind": "table",
                    "render_markdown": parts.to_markdown(),
                    "metadata": {
                        "section": current_section,
                        "title": " ".join(filter(None, [parts.title, parts.subtitle])) or "Parts list",
                        "caption": parts.subtitle,
                        "parts_list": True,
                        "verified": parts.verified,
                    },
                })

            # Step 2b: Other tables (with context)
            if camelot and not parts:
                # 'lattice' only finds ruled tables. Plenty of maintenance tables
                # (torque specs, fault codes) are whitespace-aligned with no borders,
                # which lattice misses entirely - hence the stream fallback.
                #
                # A caption like "Key to Figure 1.1" is printed on the page but is
                # never part of camelot's grid - guessed once per page (not per
                # table) since it's almost always one heading shared by whichever
                # table follows it.
                table_caption = _guess_table_caption(page_text)
                seen_tables = 0
                for flavor in ("lattice", "stream"):
                    try:
                        tables = camelot.read_pdf(file_path, pages=str(page_idx), flavor=flavor)
                    except Exception:
                        continue
                    if len(tables) == 0:
                        continue
                    for i, table in enumerate(tables):
                        df = table.df
                        ok, why = is_valid_table(df, flavor=flavor)
                        if not ok:
                            print(f"      [TableValidator] Rejected {flavor} table on page {page_idx}: {why}")
                            continue
                        title = _table_title(df, table_caption, current_section)
                        parsed_data.append({
                            "type": "table", "page": page_idx,
                            # Readable "Header: value" lines, not df.to_json().
                            # The pipeline summarises this, but keeps these rows
                            # alongside the summary so the row-level pairing
                            # ("3 = Clutch housing") survives into the embedded
                            # text instead of being paraphrased away.
                            "content": _df_to_text(df, title),
                            "kind": "table",
                            "render_markdown": _df_to_markdown(df),
                            "metadata": {
                                "section": current_section,
                                "table_index": seen_tables,
                                "flavor": flavor,
                                "caption": table_caption,
                                "title": title,
                            },
                        })
                        seen_tables += 1
                    if seen_tables:
                        break  # lattice results are cleaner; don't duplicate with stream

        # Step 3: give each figure its parts list, now that every page is read.
        self._attach_parts_lists(doc, figures, parts_lists, parsed_data,
                                 read_scanned, match_parts_list, merge_continuation,
                                 page_has_text_layer)

        # Step 4: drop running headers/footers ("Model 873 G-Series / Loader
        # Parts / 225"). The same short text on many pages says nothing about
        # any of them, yet as a near-empty chunk it matched unrelated queries -
        # it came second for a search on a part number in testing.
        return _drop_page_furniture(parsed_data, doc)

    @staticmethod
    def _attach_parts_lists(doc, figures, parts_lists, parsed_data,
                            read_scanned, match_parts_list, merge_continuation,
                            page_has_text_layer):
        """Pair every kept figure with the parts list that decodes its callouts,
        and store one searchable entry per part.

        The list is looked for on the figure's own page first, then the next two
        pages (a parts manual prints the list after the drawing), then the page
        before. On a scanned page with no text layer the list is read by vision
        and flagged unverified.
        """
        from services.page_prep import is_scanned
        from services.parts_list import drawing_family
        # Image-only pages were fully read in the main loop (scanned_page.py);
        # the older per-page scanned reader is only for pages not handled there.
        scanned_checked = {i + 1 for i in range(len(doc)) if is_scanned(doc[i])}
        list_usage = {}   # list key -> the list and every figure paired with it
        for fig in figures:
            page = fig["page"]
            chunk = fig["chunk"]
            meta = chunk["metadata"]
            numbered = bool(fig["callouts"]) or fig["layout"] == "exploded_parts"
            if not numbered:
                continue

            # Lists on the figure's page, up to two pages after and two before.
            # (A scanned manual kept its list on the page between the circuit
            # diagram and the drawings that use it.)
            window = [p for p in (page, page + 1, page + 2, page - 1, page - 2) if 1 <= p <= len(doc)]
            candidates = [parts_lists[p] for p in window if p in parts_lists]
            # Plus any list, anywhere in the manual, from the same drawing set:
            # a scanned manual printed the list for sheets CA3-15583-3 and -4 on
            # sheet CA3-15583-2.
            fams = {drawing_family(d) for d in fig.get("drawing_numbers") or [] if d}
            if fams:
                for pl in parts_lists.values():
                    if pl not in candidates and fams & {drawing_family(d) for d in pl.drawing_numbers or [] if d}:
                        candidates.append(pl)
            # Every unread scanned page in the window is read - not only when
            # nothing was found. A neighbouring figure's list already loaded (the
            # Counterweight list on the page before) used to stop the Caliper
            # Brake Kit's own list on the page after from ever being read.
            for p in window:
                if p in parts_lists or p in scanned_checked or page_has_text_layer(doc[p - 1]):
                    continue
                scanned_checked.add(p)
                pl = read_scanned(doc[p - 1])
                if pl:
                    parts_lists[p] = pl
                    candidates.append(pl)
                    # The scanned list is also stored as a table, with a
                    # "Reading" column saying how far each row is confirmed.
                    parsed_data.append({
                        "type": "table", "page": p,
                        "content": pl.to_text(),
                        "kind": "table",
                        "render_markdown": pl.to_markdown(),
                        "metadata": {
                            "title": " ".join(filter(None, [pl.title, pl.subtitle])) or "Parts list",
                            "caption": pl.subtitle,
                            "parts_list": True,
                            "verified": False,
                        },
                    })

            match = match_parts_list(fig["page_text"], fig["callouts"], candidates,
                                     figure_page=page, figure_drawings=fig.get("drawing_numbers") or [])
            if not match:
                print(f"      [PartsList] No parts list found for the figure on page {page}.")
                continue
            found, coverage, missing = match
            following = [parts_lists[p] for p in range(found.page + 1, found.page + 3) if p in parts_lists]
            plist = merge_continuation(copy.deepcopy(found), following)

            # Only the parts this drawing actually shows. One scanned list
            # (items 1-124) served three drawings; giving each drawing all 124
            # parts described parts that are not on it at all. When callouts were
            # read from the drawing, the figure gets just those.
            groups = plist.grouped()
            shown = fig["callouts"]
            if shown and len(shown) >= 3:
                groups = [(ref, rows) for ref, rows in groups if not ref or ref in shown]

            print(f"      [PartsList] Figure p{page} <- list p{plist.pages}: "
                  f"{len([g for g in groups if g[0]])} of {len(plist.refs)} parts shown on this drawing, "
                  f"callout coverage {coverage:.0%}"
                  + (f", callouts with no list entry: {missing}" if missing else ""))

            meta["parts_list"] = {
                "title": plist.title,
                "subtitle": plist.subtitle,
                "pages": plist.pages,
                "verified": plist.verified,
                "coverage": coverage,
                "missing_callouts": missing,
                "groups": [{"ref": ref, "rows": [vars(r) for r in rows]} for ref, rows in groups],
            }
            key = (plist.page, plist.title, plist.subtitle)
            usage = list_usage.setdefault(key, {"plist": plist, "figures": []})
            usage["figures"].append((page, chunk, meta.get("drawing_code"),
                                     set(shown) if shown and len(shown) >= 3 else set(plist.refs)))

        # One entry per part, per list - written once, after every figure has
        # been paired, naming only the drawing(s) the callout was actually read
        # on. Written per figure before, the 124-item scanned list became 372
        # entries, two of every three claiming a part was on a drawing that does
        # not show it.
        for usage in list_usage.values():
            plist = usage["plist"]
            heading = (" ".join(filter(None, [plist.title, plist.subtitle]))
                       or f"Parts list, page {plist.page}")
            for ref, rows in plist.grouped():
                if not ref:
                    continue
                on = [(pg, ch, code) for pg, ch, code, refs in usage["figures"] if ref in refs]
                first, alts = rows[0], rows[1:]
                text = (f"{heading} — part ref {ref}: {first.description}, part number "
                        f"{first.part_number}")
                if first.qty:
                    text += f", quantity {first.qty}"
                if first.remarks:
                    text += f", {first.remarks}"
                for a in alts:
                    text += f". Alternative: {a.description} {a.part_number}"
                    if a.qty:
                        text += f", quantity {a.qty}"
                    if a.remarks:
                        text += f", {a.remarks}"
                if on:
                    where = "; ".join(f"page {pg}" + (f" (drawing {code})" if code else "") for pg, _, code in on)
                    text += f". Shown as callout {ref} in the figure on {where}; listed on page {plist.page}."
                else:
                    text += (f". Listed on page {plist.page}; callout {ref} was not read on any drawing "
                             f"(it may be missing from the scan or too small to read).")
                if not plist.verified:
                    unsure = [r.check for r in rows if not r.check.startswith("confirmed")]
                    if not unsure:
                        text += " (Scanned page: item number, part code and quantity confirmed by two independent readings.)"
                    else:
                        text += " (Scanned page - not fully confirmed: " + " | ".join(dict.fromkeys(unsure)) + ")"
                parsed_data.append({
                    "type": "part",
                    "content": text,
                    "page": on[0][0] if on else plist.page,
                    "path": None,
                    "parent_path": on[0][1]["path"] if on else None,
                    "kind": "part",
                    "metadata": {
                        "ref": ref,
                        "part_numbers": [r.part_number for r in rows if r.part_number],
                        "parts_list_page": plist.page,
                        "verified": plist.verified,
                    },
                })
