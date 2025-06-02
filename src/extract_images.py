# src/extract_images.py

import fitz  # PyMuPDF
import os
from PIL import Image
import numpy as np
import cv2

def find_bounding_boxes(cv_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Relaxed‐rectangle version: any contour above a minimum area yields a bounding box.
    1) Build an edge mask (Canny) + variance mask (Laplacian).
    2) Combine & close small gaps.
    3) Find all contours; for each contour with area >= 0.2% of image, take its boundingRect.
    4) Filter out tiny or ultra‐thin rectangles.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) Edge mask
    edges = cv2.Canny(gray, 50, 150)
    # 2) Variance mask
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    if lap_abs.max() > 0:
        lap_norm = np.uint8((lap_abs / lap_abs.max()) * 255)
    else:
        lap_norm = np.zeros_like(gray, dtype=np.uint8)
    var_thresh = 30
    _, var_mask = cv2.threshold(lap_norm, var_thresh, 255, cv2.THRESH_BINARY)

    # 3) Combine & close
    combined = cv2.bitwise_or(edges, var_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 4) Contour detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = h * w
    min_area = 0.10 * img_area  # 0.2% of image area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Filter out very small widths/heights
        if bw < 30 or bh < 30:
            continue

        # Filter out ultra-thin slivers
        aspect = bw / float(bh)
        if aspect < 0.2 or aspect > 10.0:
            continue

        boxes.append((x, y, bw, bh))

    return boxes


def extract_images_from_pdf(pdf_path: str, output_dir: str):
    """
    1) Extract raw images from PDF (no initial crop).
    2) For each raw image, run find_bounding_boxes() on the full image.
    3) If find_bounding_boxes() returns [], treat the full image as one box.
    4) Save each box‐crop under output_dir/bounding_boxes/.
    """
    os.makedirs(output_dir, exist_ok=True)
    bbox_dir = os.path.join(output_dir, "bounding_boxes")
    os.makedirs(bbox_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base["ext"]
            filename = f"page{page_num+1}_img{img_index+1}.{ext}"
            out_path = os.path.join(output_dir, filename)

            # 1) Save the raw image bytes
            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(base["image"])

            # 2) Load it into OpenCV
            try:
                pil_img = Image.open(out_path).convert("RGB")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"WARNING: Could not load {filename}: {e}")
                continue

            # 3) Find bounding boxes
            boxes = find_bounding_boxes(cv_img)

            # 4) If no sub‐boxes found, use the whole image
            if not boxes:
                H, W = cv_img.shape[:2]
                boxes = [(0, 0, W, H)]

            # 5) Save each detected box
            for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
                try:
                    crop = pil_img.crop((x, y, x + bw, y + bh))
                    box_name = f"{os.path.splitext(filename)[0]}_box{idx}.{ext}"
                    crop.save(os.path.join(bbox_dir, box_name))
                except Exception as e:
                    print(f"WARNING: Failed to save {filename}_box{idx}: {e}")

    print(f"Raw images saved to '{output_dir}'.")
    print(f"Bounding‐box crops saved under '{bbox_dir}'.")
