import os
from src.extract_images import extract_images_from_pdf
from src.filter_with_clip import filter_with_clip
#from src.analyze_skin import segment_skin, remove_lesions, compute_skin_lab

def main():
    pdf_path = r"data/textbook1.pdf"
    extracted_dir = "data/extracted_images"
    photos_dir = "data/only_photos"
    illus_dir = "data/only_illustrations"
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Extract all extracted_images from PDF
    # extract_images_from_pdf(pdf_path, extracted_dir)

    # 2. Filter extracted_images into photos vs. illustrations

    # Adjust thresholds if needed
    filter_with_clip(
        input_folder=extracted_dir,
        photo_folder=photos_dir,
        illus_folder=illus_dir,
        labels=[
            "an image containing an unedited clinical photograph of a body, body part (hand, foot, leg, arm, face), or patch of skin containing human skin, possibly accompanied with a text description or diagram",
            "only a hand‑drawn, schematic, or computer‑generated illustration from a medical textbook, possibly with accompanying text but without any real photographs",
            "a page or portion of a page of a medical textbook containing only text, graphs, tables, and/or charts"
        ],
        margin=0.05
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
