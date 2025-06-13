# src/extract_images.py

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from paddleocr import TextDetection

# Initialize DBNet-only detector
detector = TextDetection()


def maybe_resize_cv(cv_img: np.ndarray, min_dim: int = 1024) -> tuple[np.ndarray, float]:
    h, w = cv_img.shape[:2]
    small = min(h, w)
    if small < min_dim:
        scale = min_dim / float(small)
        resized = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return resized, scale
    return cv_img, 1.0


def remove_text_with_white_boxes_cv(cv_img: np.ndarray,
                                    boxes: list[tuple[int, int, int, int]],
                                    dilate_pixels: int = 15) -> np.ndarray:
    out = cv_img.copy()
    h_img, w_img = out.shape[:2]
    for x, y, w, h in boxes:
        x0 = max(0, x - dilate_pixels)
        y0 = max(0, y - dilate_pixels)
        x1 = min(w_img, x + w + dilate_pixels)
        y1 = min(h_img, y + h + dilate_pixels)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1)
    return out


def find_bounding_boxes(cv_img: np.ndarray, min_area_frac: float = 0.005) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = cv2.Canny(gray, 50, 150)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    if lap_abs.max() > 0:
        lap_norm = (lap_abs / lap_abs.max() * 255).astype(np.uint8)
    else:
        lap_norm = np.zeros_like(gray, dtype=np.uint8)
    _, var_mask = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(edges, var_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    min_area = min_area_frac * (h * w)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 30 or bh < 30:
            continue
        aspect = bw / float(bh)
        if aspect < 0.2 or aspect > 10.0:
            continue
        boxes.append((x, y, bw, bh))
    return boxes


def extract_images_from_pdf(
        pdf_path: str,
        output_dir: str,
        ocr_score_thresh: float = 0.85,
        dilate_pixels: int = 15,
        min_resize_dim: int = 1024,
        use_ocr: bool = True
):
    """
    1) Render all pages to raw_dir.
    2) Mask vector text on each raw page → mask1_dir.
    3) If use_ocr:
         OCR-mask each mask1 image → mask2_dir.
       Else:
         skip OCR, use mask1_dir as mask2_dir.
    4) Find & crop bounding boxes from each mask2 image → crop_dir.
    """
    # prepare output dirs
    raw_dir   = os.path.join(output_dir, "page_images")
    mask1_dir = os.path.join(output_dir, "page_images_masked_pymupdf")
    mask2_dir = os.path.join(output_dir, "page_images_masked_ocr")
    crop_dir  = os.path.join(output_dir, "bounding_boxes")
    for d in (raw_dir, mask1_dir, crop_dir):
        os.makedirs(d, exist_ok=True)
    if use_ocr:
        os.makedirs(mask2_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    # 1) Render all pages
    for idx, page in enumerate(doc, start=1):
        pix      = page.get_pixmap(dpi=300)
        page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_img.save(os.path.join(raw_dir, f"page{idx:03d}.png"))

    # 2) PyMuPDF text-mask on raw images
    for fname in sorted(os.listdir(raw_dir)):
        idx       = int(fname[4:7])
        page      = doc[idx - 1]
        cv_page   = cv2.cvtColor(np.array(Image.open(os.path.join(raw_dir, fname))), cv2.COLOR_RGB2BGR)

        words     = page.get_text("words")
        sx = pix.width  / page.rect.width
        sy = pix.height / page.rect.height
        word_boxes = [
            (int(x0 * sx), int(y0 * sy), int((x1 - x0) * sx), int((y1 - y0) * sy))
            for (x0, y0, x1, y1, *_) in words
        ]

        cv_masked1 = remove_text_with_white_boxes_cv(cv_page, word_boxes, dilate_pixels)
        Image.fromarray(cv2.cvtColor(cv_masked1, cv2.COLOR_BGR2RGB)) \
             .save(os.path.join(mask1_dir, fname))

    # 3) OCR-mask on mask1 images (optional)
    if use_ocr:
        for fname in sorted(os.listdir(mask1_dir)):
            cv_img = cv2.cvtColor(np.array(Image.open(os.path.join(mask1_dir, fname))), cv2.COLOR_RGB2BGR)
            ocr_img, scale = maybe_resize_cv(cv_img, min_dim=min_resize_dim)

            results = detector.predict(input=ocr_img)
            boxes = []
            for det in results:
                for quad, score in zip(det["dt_polys"], det["dt_scores"]):
                    if score < ocr_score_thresh:
                        continue
                    xs, ys = quad[:, 0], quad[:, 1]
                    x0, y0 = int(xs.min()), int(ys.min())
                    w, h   = int(xs.max() - x0), int(ys.max() - y0)
                    boxes.append((int(x0 / scale), int(y0 / scale), int(w / scale), int(h / scale)))

            cv_masked2 = remove_text_with_white_boxes_cv(cv_img, boxes, dilate_pixels)
            Image.fromarray(cv2.cvtColor(cv_masked2, cv2.COLOR_BGR2RGB)) \
                 .save(os.path.join(mask2_dir, fname))
    else:
        # skip OCR: point mask2_dir at mask1_dir
        mask2_dir = mask1_dir

    # 4) Bounding-box crop on mask2 images
    for fname in sorted(os.listdir(mask2_dir)):
        base = os.path.splitext(fname)[0]
        cv_img = cv2.cvtColor(np.array(Image.open(os.path.join(mask2_dir, fname))), cv2.COLOR_RGB2BGR)

        regions = find_bounding_boxes(cv_img, min_area_frac=0.005)
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        for i, (x, y, w, h) in enumerate(regions, start=1):
            crop = pil_img.crop((x, y, x + w, y + h))
            cv_crop, _ = maybe_resize_cv(cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR),
                                         min_dim=min_resize_dim)
            out = Image.fromarray(cv2.cvtColor(cv_crop, cv2.COLOR_BGR2RGB))
            out.save(os.path.join(crop_dir, f"{base}_box{i:02d}.png"))

    print(f"Saved raw → '{raw_dir}'")
    print(f"Saved pymupdf-masked → '{mask1_dir}'")
    if use_ocr:
        print(f"Saved ocr-masked → '{mask2_dir}'")
    print(f"Saved crops → '{crop_dir}'")
