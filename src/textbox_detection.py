# src/textbox_detection.py

import cv2
import os
import shutil


def is_textbox_image(
        image_path: str,
        threshold: float = 0.5,
        window_size: int = 10
) -> bool:
    """
    Return True if a bounding‐box image is predominantly one “band” of grayscale.
    1) Read image in BGR, convert to gray.
    2) Compute 256‑bin histogram.
    3) Find index of the histogram peak (i_peak).
    4) Sum counts in [i_peak - window_size ... i_peak + window_size].
    5) If that sum / total_pixels >= threshold → textbox.

    Args:
      image_path: path to a single bounding‐box image.
      threshold: fraction of total pixels required (e.g. 0.5).
      window_size: half‑width around the peak gray level (e.g. 10).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    total_pixels = h * w

    # Compute 256‑bin histogram of grayscale intensities
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Locate peak gray level
    i_peak = int(hist.argmax())

    # Determine the inclusive range around i_peak
    start_bin = max(0, i_peak - window_size)
    end_bin = min(255, i_peak + window_size)

    # Sum up all pixels whose gray level is within [start_bin..end_bin]
    band_count = hist[start_bin: end_bin + 1].sum()

    # If this band contains ≥ threshold fraction of pixels, treat as textbox
    return (band_count / total_pixels) >= threshold


def filter_textboxes(bbox_dir: str, threshold: float = 0.5, window_size: int = 10):
    """
    Iterate through every file in 'bbox_dir'. Classify each via is_textbox_image():
      • If True → move into 'bbox_dir/textboxes/'.
      • Else   → move into 'bbox_dir/non_textboxes/'.

    Creates subfolders if needed.
    """
    text_dir = os.path.join(bbox_dir, "textboxes")
    nontext_dir = os.path.join(bbox_dir, "non_textboxes")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(nontext_dir, exist_ok=True)

    for fname in os.listdir(bbox_dir):
        src_path = os.path.join(bbox_dir, fname)
        if not os.path.isfile(src_path):
            continue
        # Only process image files
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue

        try:
            if is_textbox_image(src_path, threshold=threshold, window_size=window_size):
                shutil.move(src_path, os.path.join(text_dir, fname))
            else:
                shutil.move(src_path, os.path.join(nontext_dir, fname))
        except Exception as e:
            print(f"WARNING: Could not classify '{fname}': {e}")

    print(f"Textboxes → '{text_dir}'")
    print(f"Non‑textboxes → '{nontext_dir}'")
