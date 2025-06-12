# src/extract_images.py

import fitz  # PyMuPDF
import os
from PIL import Image
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────────────────────
# 1) TEXT DETECTION HELPERS (EAST)
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_boxes_east(
    image: np.ndarray,
    net: cv2.dnn_Net,
    min_confidence: float = 0.3,
    width: int = 768,
    height: int = 768
) -> list[tuple[int, int, int, int]]:
    """
    Run the EAST text detector on `image` (resized internally to width×height),
    returning a list of bounding boxes (x, y, w, h) where text is detected.
    """
    orig_h, orig_w = image.shape[:2]
    rW = orig_w / float(width)
    rH = orig_h / float(height)

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (width, height),
        (123.68, 116.78, 103.94),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0_data = geometry[0, 0, y]
        x1_data = geometry[0, 1, y]
        x2_data = geometry[0, 2, y]
        x3_data = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = float(angles_data[x])
            cosA = np.cos(angle)
            sinA = np.sin(angle)

            h = float(x0_data[x] + x2_data[x])
            w = float(x1_data[x] + x3_data[x])
            endX = int(offsetX + (cosA * x1_data[x]) + (sinA * x2_data[x]))
            endY = int(offsetY - (sinA * x1_data[x]) + (cosA * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Scale back to original image size
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            rects.append((startX, startY, endX - startX, endY - startY))
            confidences.append(float(scores_data[x]))

    # Perform non-maximum suppression to filter overlapping boxes
    idxs = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
    final_boxes = []
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten()
    for i in idxs:
        final_boxes.append(rects[i])

    return final_boxes


def remove_text_with_white_boxes(
    image: np.ndarray,
    text_boxes: list[tuple[int, int, int, int]],
    dilate_pixels: int = 15
) -> np.ndarray:
    """
    Given `image` and a list of text bounding boxes, dilate each box by `dilate_pixels`
    and cover it with a filled white rectangle. Returns the modified image.
    """
    img_h, img_w = image.shape[:2]
    output = image.copy()

    for (x, y, w, h) in text_boxes:
        x0 = max(0, x - dilate_pixels)
        y0 = max(0, y - dilate_pixels)
        x1 = min(img_w, x + w + dilate_pixels)
        y1 = min(img_h, y + h + dilate_pixels)
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 255, 255), -1)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# 2) BOUNDING BOX DETECTION (RELAXED-RECTANGLE)
# ─────────────────────────────────────────────────────────────────────────────

def find_bounding_boxes(cv_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Given a BGR image, build an edge mask (Canny) + variance mask (Laplacian),
    combine them and close gaps, then find contours. For each contour whose area
    >= 10% of the image area, return its bounding rectangle (x, y, w, h), filtering
    out very small or ultra-thin regions.
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
    _, var_mask = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)

    # 3) Combine & close small gaps
    combined = cv2.bitwise_or(edges, var_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 4) Contour detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_area = h * w
    min_area = 0.10 * img_area  # 10% of image area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue

        aspect = bw / float(bh)
        if aspect < 0.2 or aspect > 10.0:
            continue

        boxes.append((x, y, bw, bh))

    return boxes


# ─────────────────────────────────────────────────────────────────────────────
# 3) MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(
    pdf_path: str,
    output_dir: str,
    east_model_path: str,
    remove_text: bool = True,
    min_confidence: float = 0.3,
    east_width: int = 768,
    east_height: int = 768,
    dilate_pixels: int = 15
):
    """
    1) Extract all raw images from the PDF at pdf_path into output_dir.
    2) For each extracted image:
       a) Load as OpenCV BGR.
       b) If remove_text=True:
            • Use EAST to detect text boxes (min_confidence, size=east_width×east_height).
            • Cover text with white boxes (dilation = dilate_pixels).
          Else:
            • Skip text detection/removal.
       c) Find bounding boxes on the (possibly white-boxed) image.
       d) If no boxes, use the full image as a single box.
       e) Save each box cropped from the processed image to output_dir/bounding_boxes/.
    """
    os.makedirs(output_dir, exist_ok=True)
    bbox_dir = os.path.join(output_dir, "bounding_boxes")
    os.makedirs(bbox_dir, exist_ok=True)

    # Load the EAST text detector (frozen graph) once
    if remove_text:
        net = cv2.dnn.readNet(east_model_path)
    else:
        net = None

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base["ext"]
            filename = f"page{page_num+1}_img{img_index+1}.{ext}"
            out_path = os.path.join(output_dir, filename)

            # --- Step 1: Save raw image bytes if not already present ---
            if not os.path.exists(out_path):
                with open(out_path, "wb") as f:
                    f.write(base["image"])

            # --- Step 2: Load image into OpenCV (BGR) ---
            try:
                pil_img = Image.open(out_path).convert("RGB")
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"WARNING: Could not load {filename}: {e}")
                continue

            # --- Step 3: Optionally detect text and cover with white boxes ---
            if remove_text:
                text_boxes = detect_text_boxes_east(
                    cv_img, net,
                    min_confidence=min_confidence,
                    width=east_width, height=east_height
                )

                if text_boxes:
                    cv_img_ntext = remove_text_with_white_boxes(cv_img, text_boxes, dilate_pixels=dilate_pixels)
                else:
                    # No text detected; proceed with original image
                    cv_img_ntext = cv_img.copy()
            else:
                # Skip text removal entirely
                cv_img_ntext = cv_img.copy()

            # Convert the processed OpenCV image back to a PIL image
            pil_processed = Image.fromarray(cv2.cvtColor(cv_img_ntext, cv2.COLOR_BGR2RGB))

            # --- Step 4: Find bounding boxes on the processed image ---
            boxes = find_bounding_boxes(cv_img_ntext)

            # --- Step 5: If no boxes found, use the full image as a fallback ---
            if not boxes:
                H, W = cv_img_ntext.shape[:2]
                boxes = [(0, 0, W, H)]

            # --- Step 6: Save each detected box (cropped from the processed image) ---
            for idx, (x, y, bw, bh) in enumerate(boxes, start=1):
                try:
                    box_crop = pil_processed.crop((x, y, x + bw, y + bh))
                    box_name = f"{os.path.splitext(filename)[0]}_box{idx}.{ext}"
                    box_path = os.path.join(bbox_dir, box_name)
                    box_crop.save(box_path)
                except Exception as e:
                    print(f"WARNING: Failed to save {filename}_box{idx}: {e}")

    print(f"Raw images saved to '{output_dir}'.")
    print(f"Bounding-box crops saved under '{bbox_dir}'.")
