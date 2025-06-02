from src.filter_with_clip import get_image_label_similarities
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
        "a photograph", "a photograph accompanied with a text description"
    ]

illus_labels = [
        "only an illustration",
        "only a portion of text",
        "only a blank page"
    ]

scores = get_image_label_similarities("data/extracted_images/bounding_boxes/page33_img7_box1.jpeg", photo_labels, illus_labels)
print("Photo‐label similarities:")
for lbl, score in scores["photo"].items():
    print(f"{lbl:60} → {score:.4f}")

print("\nIllustration‐label similarities:")
for lbl, score in scores["illus"].items():
    print(f"{lbl:60} → {score:.4f}")
