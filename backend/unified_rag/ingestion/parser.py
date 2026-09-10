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
                        
                        parent_ctx = f"Figure on Page {page_idx} under section '{current_section}'"
                        print(f"      [Figure] Decomposing composite drawing...")
                        
                        try:
                            sub_figures = splitter.split_image_sam(raw_crop, parent_context=parent_ctx)
                        except Exception as e:
                            print(f"      ⚠️ [Parser] Figure decomposition crashed: {e}. Using fallback.")
                            sub_figures = []
                        
                        if not sub_figures:
                            # Fallback to simple crop if splitter finds nothing or crashes.
                            # Upload directly from memory (cv2.imencode) — no local file at all.
                            _, buf = cv2.imencode(".png", raw_crop)
                            cloud_url = self.cloudinary.upload_image(
                                io.BytesIO(buf.tobytes()), f"{manual_id}_p{page_idx}_fig{img_index}"
                            )

                            if cloud_url:
                                parsed_data.append({
                                    "type": "image", "path": cloud_url, "page": page_idx,
                                    "metadata": {"section": current_section, "label": "Main Diagram"}
                                })
                            else:
                                print(f"      ⚠️ [Parser] Cloudinary upload unavailable — skipping image chunk (page {page_idx}, fig {img_index}).")
                            img_index += 1
                        else:
                            for i, sub in enumerate(sub_figures):
                                _, buf = cv2.imencode(".png", sub["crop"])
                                cloud_url = self.cloudinary.upload_image(
                                    io.BytesIO(buf.tobytes()), f"{manual_id}_p{page_idx}_sub{img_index}_{i}"
                                )

                                if cloud_url:
                                    parsed_data.append({
                                        "type": "image", "path": cloud_url, "page": page_idx,
                                        "metadata": {
                                            "section": current_section,
                                            "label": sub["label"],
                                            "parent_context": parent_ctx
                                        }
                                    })
                                    print(f"         ∟ Isolated component: {sub['label']}")
                                else:
                                    print(f"      ⚠️ [Parser] Cloudinary upload unavailable — skipping sub-figure chunk ({sub['label']}).")
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
                try:
                    tables = camelot.read_pdf(file_path, pages=str(page_idx), flavor='lattice')
                    for i, table in enumerate(tables):
                        parsed_data.append({
                            "type": "table", "page": page_idx, "content": table.df.to_json(),
                            "metadata": {"section": current_section, "table_index": i}
                        })
                except Exception: pass
            
        return parsed_data
            
        return parsed_data
