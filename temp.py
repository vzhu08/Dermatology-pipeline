from src.filter_with_clip import get_image_label_similarities
import cv2

'''
photo_labels = [
    "a real clinical photograph of a portion of human skin showing real texture or lesions",
    "a real clinical photograph of a body part with visible skin",
    "a real clinical photograph of a face with visible skin",
    "a real clinical photograph of a body with visible skin"
]
illus_labels = [
    "a 2-dimensional hand-drawn or computer-generated medical illustration or diagram",
    "a complex 3-dimensional hand-drawn or computer-generated medical model",
    "an portion of a textbook page containing only text",
    "an portion of a textbook showing a chart"
]
'''
'''photo_labels = ["a photograph",
                "a photograph of a hand",
                "a photograph of a finger",
                "a photograph of an arm",
                "a photograph of a leg",
                "a photograph of a foot",
                "a photograph of a patient's body",
                "a photograph of a face",
                "a photograph of a person",
                "a photograph of skin",
                "a photograph of a skin condition"
                ]

illus_labels = [
        "only a medical illustration",
        "only a line drawing illustration",
        "only a schematic chart",
        "only a scanned textbook table",
        "only a diagram with labels",
        "only a block of printed text",
        "only a page with some text",
        "only a blank page"
        ]
'''

#photo_labels = ["a photograph of a body part", "a photograph of a person", "a photograph of a face"]
#illus_labels = ["only an illustration", "only a chart", "only a portion of text", "only a blank page"]

photo_labels = [
        "a photograph",
        "a photograph of skin lesions",
        "a photograph of a patient’s skin",
        "a photograph of a body part",
        "a photograph of a face",
        "a photograph of a patient's back",
        "a photograph of a patient's torso"
    ]

illus_labels = [
        "only an illustration",
        "only a hand-drawn illustration",
        "only a computer-generated illustration",
        "only a portion of text",
        "only a collection of text characters",
        "only a blank page",
        "only a medical illustration line drawing",
        "only a black-and-white diagram with labels",
        "only a schematic chart or graph",
        "only a scanned textbook table of data",
        "only a stylized drawing of skin anatomy",
        "only a block of printed text or table",
        "only a few words on a page"
    ]

'''
scores = get_image_label_similarities("data/extracted_images/bounding_boxes/page11_img2_box1.jpeg", photo_labels, illus_labels)
print("Photo‐label similarities:")
for lbl, score in scores["photo"].items():
    print(f"{lbl:60} → {score:.4f}")

print("\nIllustration‐label similarities:")
for lbl, score in scores["illus"].items():
    print(f"{lbl:60} → {score:.4f}")
'''

import cv2
import numpy as np

def debug_east_on_image(image_path: str, east_model_path: str):
    """
    Loads one image and runs EAST; draws all detected text boxes in red.
    Saves a PNG with “_east_debug” appended so you can inspect them.
    """
    # 1) Load EAST
    net = cv2.dnn.readNet(east_model_path)

    # 2) Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: could not load {image_path}")
        return

    # 3) Run EAST at 512×512 (higher res for small text)
    def detect(image, min_conf=0.4, W=512, H=512):
        h0, w0 = image.shape[:2]
        rW = w0 / float(W)
        rH = h0 / float(H)
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                      (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        rects = []
        confidences = []
        for y in range(scores.shape[2]):
            for x in range(scores.shape[3]):
                if float(scores[0, 0, y, x]) < min_conf:
                    continue
                # decode geometry
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = float(geometry[0, 4, y, x])
                cosA = np.cos(angle)
                sinA = np.sin(angle)
                h = float(geometry[0, 0, y, x] + geometry[0, 2, y, x])
                w = float(geometry[0, 1, y, x] + geometry[0, 3, y, x])
                endX = int(offsetX + (cosA * geometry[0, 1, y, x]) + (sinA * geometry[0, 2, y, x]))
                endY = int(offsetY - (sinA * geometry[0, 1, y, x]) + (cosA * geometry[0, 2, y, x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # scale back up
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                rects.append((startX, startY, endX - startX, endY - startY))
                confidences.append(float(scores[0, 0, y, x]))
        # NMS
        idxs = cv2.dnn.NMSBoxes(rects, confidences, min_conf, 0.4)
        boxes = []
        if isinstance(idxs, np.ndarray):
            idxs = idxs.flatten()
        for i in idxs:
            boxes.append(rects[i])
        return boxes

    boxes = detect(img, min_conf=0.4, W=512, H=512)

    # 4) Draw them in red
    debug = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 5) Save a debug image
    outp = image_path.replace(".", "_east_debug.")
    cv2.imwrite(outp, debug)
    print(f"Debug image with EAST boxes saved to {outp}")


debug_east_on_image(
    image_path="data/extracted_images/page2_img1.jpeg",
    east_model_path="src/frozen_east_text_detection.pb"
)
