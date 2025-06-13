import os
import cv2
import layoutparser as lp

def classify_pages(input_folder: str, output_folder: str, score_thresh: float = 0.8):
    """
    Detect and annotate layout blocks on each page image in input_folder using PubLayNet model.
    Saves annotated images to output_folder.
    """
    # Load PubLayNet model via LayoutParser
    model = lp.Detectron2LayoutModel(
        config_path="models/publaynet/config.yml",
        model_path="models/publaynet/model_final.pth",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )

    # Prepare output directory
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over images in input_folder
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            continue
        input_path = os.path.join(input_folder, fname)
        image = cv2.imread(input_path)
        if image is None:
            print(f"[WARN] Could not load image: {input_path}")
            continue

        # Detect layout blocks
        layout = model.detect(image)

        # Annotate image
        annotated = image.copy()
        for block in layout:
            x0, y0, x1, y1 = map(int, block.coordinates)
            label = block.type
            # Draw bounding box
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # Put label text
            cv2.putText(
                annotated, label, (x0, max(y0 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Save annotated image
        output_path = os.path.join(output_folder, fname)
        cv2.imwrite(output_path, annotated)
        print(f"[INFO] Processed {fname}: {len(layout)} blocks detected. Saved to {output_path}")
