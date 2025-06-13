# main

import os
import torch
from src.extract_images import extract_images_from_pdf
from src.filter_with_clip import filter_with_clip


def main():
    pdf_path = "data/textbook1.pdf"
    extracted_dir = "data/extracted_images"
    sorted_images = "data/sorted_images"
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(sorted_images, exist_ok=True)

    # 1. Extract all extracted_images from PDF
    # extract_images_from_pdf(pdf_path=pdf_path, output_dir=extracted_dir, use_ocr=False)

    # 2. Filter extracted_images into photos vs. illustrations
    photo_labels = [
        "a photograph",
        "a high-resolution photo",
        "a DSLR photo",
        "a real-life photo",
        "a photo taken with a camera",
        "a snapshot"
    ]

    illus_labels = [
        "a drawing",
        "digital art",
        "a cartoon",
        "an illustration",
        "a portion of text",
        "a blank page"
    ]

    filter_with_clip(
        input_folder="data/extracted_images/bounding_boxes",
        output_folder=sorted_images,
        use_mean=True,
        categories={
        "photo": photo_labels,
        "illus": illus_labels
        },
        max_workers=10
    )

    # 2. Filter photos into skin and no skin
    skin_labels = [
        "a human",
        "human skin",
        "a skin lesion",
        "a skin condition",
        "a leg",
        "a foot",
        "an arm",
        "a hand",
        "a patient's torso",
        "a patient's back",
        "a face",
        "a mouth",
        "a tongue",
        "an ear",
        "a nose",
        "a finger"
    ]

    noskin_labels = [
        "a microscopic photo",
        "a micrograph",
        "a biological cell",
        "a group of cells",
        "a molecule",
        "medical equipment",
        "a tool",
        "a page with text",
        "a blank page",
    ]

    filter_with_clip(
        input_folder="data/sorted_images/photo",
        output_folder=sorted_images,
        categories={
        "skin": skin_labels,
        "no_skin": noskin_labels
        },
        max_workers=10
    )


    '''
    # 3. Analyze each photo
    csv_path = os.path.join(results_dir, "skin_tone_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Avg_L", "Avg_A", "Avg_B", "Fitz_Tone"])
        for img_file in os.listdir(photos_dir):
            img_path = os.path.join(photos_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            skin_mask = segment_skin(img_rgb)
            healthy_mask = remove_lesions(skin_mask, img_rgb)
            avg_L, avg_A, avg_B = compute_skin_lab(healthy_mask, img_rgb)
            tone = classify_fitzpatrick(avg_A, avg_B)  # define this in analyze_skin.py

            writer.writerow([img_file, avg_L, avg_A, avg_B, tone])

    '''
    print("Pipeline complete. Check data/results/ for outputs.")

if __name__ == "__main__":
    main()
