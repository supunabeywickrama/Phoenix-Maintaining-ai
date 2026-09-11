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
        - Uses FigureSplitter to decompose composite drawings.
        """
        from services.figure_splitter import FigureSplitter
        from services.figure_validator import is_valid_figure
        splitter = FigureSplitter()
        
        print(f"📄 [Parser] Opening PDF: {file_path}")
        doc = fitz.open(file_path)
        parsed_data = []
        total_pages = len(doc)
        
        # In-memory tracking of the document's structural hierarchy
        current_section = "General Information"
        
        print(f"📖 [Parser] PDF has {total_pages} pages. Starting extraction...")
        
        for page_num in range(total_pages):
            page_idx = page_num + 1
            print(f"   ∟ Processing Page {page_idx}/{total_pages}...")
            page = doc.load_page(page_num)

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

                    # FIGURES: Use Agentic Splitting
                    if "picture" in class_name or "figure" in class_name:
                        # Extract the raw region
                        raw_crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
                        
                        # Validate BEFORE spending vision calls: the layout model's
                        # 'Picture' class also catches logos, warning icons, rules and
                        # dense text blocks, which are useless as retrievable figures.
                        keep, why = is_valid_figure(raw_crop)
                        if not keep:
                            print(f"      🚫 [Validator] Rejected region on page {page_idx}: {why}")
                            continue

                        parent_ctx = f"Figure on Page {page_idx} under section '{current_section}'"
                        print(f"      [Figure] Accepted ({why}). Decomposing composite drawing...")
                        
                        try:
                            sub_figures = splitter.split_image_sam(raw_crop, parent_context=parent_ctx)
                        except Exception as e:
                            print(f"      ⚠️ [Parser] Figure decomposition crashed: {e}. Using fallback.")
                            sub_figures = []
                        
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
                            parsed_data.append({
                                "type": "image", "path": full_url, "page": page_idx,
                                "figure_role": "full", "parent_path": None,
                                "width": fw, "height": fh,
                                "metadata": {
                                    "section": current_section,
                                    "label": "Full Diagram",
                                    "figure_role": "full",
                                    "page_text": page_text,
                                },
                            })
                        else:
                            print(f"      [Parser] Upload unavailable - skipping figure (page {page_idx}).")

                        for i, sub in enumerate(sub_figures):
                            sub_keep, sub_why = is_valid_figure(sub["crop"], use_vision=False)
                            if not sub_keep:
                                print(f"         [Validator] Dropped component: {sub_why}")
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

            # Step 2: Tables (with context)
            if camelot:
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
            
        return parsed_data
            
        return parsed_data
