import fitz  # PyMuPDF
import io
import logging
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
                        if text_content:
                            parsed_data.append({
                                "type": "text", "content": text_content, "page": page_idx,
                                "metadata": {"section": current_section}
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
                        parsed_data.append({
                            "type": "table", "page": page_idx,
                            "content": df.to_json(),
                            "kind": "table",
                            "render_markdown": _df_to_markdown(df),
                            "metadata": {
                                "section": current_section,
                                "table_index": seen_tables,
                                "flavor": flavor,
                            },
                        })
                        seen_tables += 1
                    if seen_tables:
                        break  # lattice results are cleaner; don't duplicate with stream
            
        return parsed_data
            
        return parsed_data
