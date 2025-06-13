from src.filter_with_clip import get_image_label_similarities
from src.layout_detection import classify_pages
import layoutparser as lp

classify_pages(input_folder="data/extracted_images/page_images", output_folder="data/layouts")
skin_labels = [
    "a human",
    "human skin",
    "photograph of a skin lesion",
    "photograph of a skin condition",
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

'''
scores = get_image_label_similarities(
        image_path="data/extracted_images/bounding_boxes/page129_box01.png",
        categories={
        "skin": skin_labels,
        "no_skin": noskin_labels
        })
print(scores["label_scores"])
print(scores["classification"])

'''

def visualize_text_detection_boxes(
    image_path: str,
    output_path: str,
    ocr_score_thresh: float = 0.0
):
    """
    Runs PaddleOCR’s DBNet text detector on a single image, draws each
    polygon’s bounding box and confidence score, and saves the result.

    image_path:        path to the input image
    output_path:       where to write the annotated image
    ocr_score_thresh:  only draw boxes with score >= this
    """
    import cv2
    from paddleocr import TextDetection

    # 1) Initialize detector
    detector = TextDetection()

    # 2) Load image (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    # 3) Detect text polygons
    results = detector.predict(input=img)

    # 4) Draw each polygon and its confidence
    for det in results:
        polys  = det["dt_polys"]   # shape (N,4,2)
        scores = det["dt_scores"]  # list of N floats
        for quad, score in zip(polys, scores):
            if score < ocr_score_thresh:
                continue
            pts = quad.astype(int)
            # draw the polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
            # label with confidence
            x, y = pts[:,0].min(), pts[:,1].min()
            cv2.putText(
                img,
                f"{score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA
            )

    # 5) Write out annotated image
    cv2.imwrite(output_path, img)

'''visualize_text_detection_boxes(
    image_path="data/extracted_images/page_images_masked_pymupdf/page025.png",
    output_path="data/debug/page001_ocr_vis.png",
    ocr_score_thresh=0.6
)'''
