# src/extract_images.py

import fitz  # PyMuPDF
import os


def extract_images_from_pdf(pdf_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_ext = base_image["ext"]
            filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
            out_path = os.path.join(output_dir, filename)

            if os.path.exists(out_path):
                # skip if it’s already there
                continue

            with open(out_path, "wb") as f:
                f.write(base_image["image"])

    print(f"Extracted images saved to {output_dir}")
