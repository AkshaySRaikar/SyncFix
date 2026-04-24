# ingestion/pdf_extractor.py
import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import Generator
import io
from PIL import Image as PILImage

def extract_text_and_images(
    pdf_path: str,
    image_output_dir: str,
    min_image_size: int = 50
) -> Generator[dict, None, None]:
    """
    Extract text blocks and images from each PDF page.
    Yields a dict per page with text and saved image paths.
    """
    Path(image_output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem

    for page_num, page in enumerate(doc):
        # --- Text ---
        text = page.get_text("text").strip()

        # --- Images ---
        image_paths = []
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]

            try:
                pil_img = PILImage.open(io.BytesIO(img_bytes))
                w, h = pil_img.size
                if w < 100 or h < 100:   # skip tiny icons/decorations
                    continue
            except Exception:
                continue   

            fname = f"{pdf_name}_p{page_num}_img{img_idx}.{ext}"
            fpath = os.path.join(image_output_dir, fname)
            with open(fpath, "wb") as f:
                f.write(img_bytes)
            image_paths.append(fpath)

        yield {
            "page": page_num,
            "pdf": pdf_name,
            "text": text,
            "image_paths": image_paths,
        }

    doc.close()