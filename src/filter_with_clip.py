# src/clip_filter.py

import os
import glob
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

def load_clip_model(device: str = None):
    """
    Load CLIP model, tokenizer, and image processor.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, device

def is_photograph_clip(
    image_path: str,
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    image_processor: CLIPImageProcessor,
    device: str,
    labels: list[str],
    margin: float = 0.05
) -> bool:
    """
    Zero‑shot classification with CLIP, requiring a margin to avoid ambiguous cases.

    Returns True if the image at image_path is classified as the first label (photo)
    with at least `margin` difference above the next-highest label score.
    """
    # 1) Load image and convert to RGB
    pil_image = Image.open(image_path).convert("RGB")

    # 2) Tokenize text prompts
    text_inputs = tokenizer(
        labels,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 3) Preprocess image via CLIPImageProcessor
    image_inputs = image_processor(
        images=pil_image,
        return_tensors="pt"
    ).to(device)

    # 4) Forward pass through CLIP
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs["pixel_values"]
        )
        image_embeds = outputs.image_embeds    # shape (1, D)
        text_embeds  = outputs.text_embeds     # shape (len(labels), D)

    # 5) Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # 6) Compute cosine similarities
    sims = (image_embeds @ text_embeds.T).squeeze(0).cpu().numpy()

    # 7) Determine top two indices
    top_idx = int(np.argmax(sims))
    sims_copy = sims.copy()
    sims_copy[top_idx] = -np.inf
    second_idx = int(np.argmax(sims_copy))

    # 8) Check margin
    if top_idx == 0 and (sims[top_idx] - sims[second_idx] >= margin):
        return True
    return False

def filter_with_clip(
    input_folder: str,
    photo_folder: str,
    illus_folder: str,
    labels: list[str],
    margin: float = 0.05
):
    """
    Walk through all images in input_folder and sort them:
      • if is_photograph_clip(...) == True → copy to photo_folder
      • else → copy to illus_folder

    Clears out any existing files in photo_folder and illus_folder before processing.
    """
    # 1) Create/clear output directories
    os.makedirs(photo_folder, exist_ok=True)
    os.makedirs(illus_folder, exist_ok=True)
    for folder in (photo_folder, illus_folder):
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # 2) Load CLIP model components once
    model, tokenizer, image_processor, device = load_clip_model()

    # 3) Gather all image file paths in input_folder
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.jpeg"]
    all_files = []
    for pat in patterns:
        all_files += glob.glob(os.path.join(input_folder, pat))

    # 4) Process each image
    for img_path in all_files:
        try:
            is_photo = is_photograph_clip(
                image_path=img_path,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                device=device,
                labels=labels,
                margin=margin
            )
        except Exception as e:
            print(f"WARNING: Failed to process {img_path}: {e}")
            is_photo = False

        filename = os.path.basename(img_path)
        dest = photo_folder if is_photo else illus_folder

        # Copy by re‑loading with PIL (no extra dependencies required)
        try:
            img = Image.open(img_path)
            img.save(os.path.join(dest, filename))
        except Exception as e:
            print(f"WARNING: Could not save {img_path} to {dest}: {e}")

    print(f"CLIP filtering done. Photos → '{photo_folder}', Illustrations → '{illus_folder}'.")
