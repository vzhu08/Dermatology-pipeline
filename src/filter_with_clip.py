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
    Returns (model, tokenizer, image_processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    return model, tokenizer, image_processor, device

def get_image_label_similarities(
    image_path: str,
    photo_labels: list[str],
    illus_labels: list[str]
) -> dict[str, dict[str, float]]:
    """
    Compute cosine similarities between a single image and two sets of text labels:
    - photo_labels: prompts describing photos
    - illus_labels: prompts describing illustrations

    Returns a dictionary:
      {
        "photo":   { photo_label1: score1, photo_label2: score2, ... },
        "illus":   { illus_label1: score1, illus_label2: score2, ... }
      }
    """
    model, tokenizer, image_processor, device = load_clip_model()

    pil_image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(
        images=pil_image,
        return_tensors="pt"
    ).to(device)

    all_labels = photo_labels + illus_labels
    text_inputs = tokenizer(
        all_labels,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs["pixel_values"]
        )
        image_embeds = outputs.image_embeds    # shape: (1, D)
        text_embeds  = outputs.text_embeds     # shape: (len(all_labels), D)

    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

    sims = (image_embeds @ text_embeds.T).squeeze(0).cpu().numpy()  # shape: (len(all_labels),)

    photo_scores = {
        photo_labels[i]: float(sims[i])
        for i in range(len(photo_labels))
    }
    illus_scores = {
        illus_labels[j]: float(sims[len(photo_labels) + j])
        for j in range(len(illus_labels))
    }

    return {"photo": photo_scores, "illus": illus_scores}

def is_photo_by_labels(
    image_path: str,
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    image_processor: CLIPImageProcessor,
    device: str,
    photo_labels: list[str],
    illus_labels: list[str]
) -> bool:
    """
    Return True if the image’s highest cosine score among photo_labels
    exceeds the highest among illus_labels. Otherwise return False.
    """
    pil_image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

    all_labels = photo_labels + illus_labels
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs["pixel_values"]
        )
        image_embeds = outputs.image_embeds    # (1, D)
        text_embeds  = outputs.text_embeds     # (len(all_labels), D)

    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)

    sims = (image_embeds @ text_embeds.T).squeeze(0).cpu().numpy()  # (len(all_labels),)

    max_photo = max(sims[:len(photo_labels)]) if photo_labels else float("-inf")
    max_illus = max(sims[len(photo_labels):]) if illus_labels else float("-inf")

    return max_photo > max_illus

def filter_with_clip_two_label_lists(
    input_folder: str,
    photo_folder: str,
    illus_folder: str,
    photo_labels: list[str],
    illus_labels: list[str]
):
    """
    Walk through all images in input_folder and sort them:
      • If max(photo_label_scores) > max(illus_label_scores) → save to photo_folder
      • Else → save to illus_folder

    Clears out any existing files in photo_folder and illus_folder before processing.
    """
    os.makedirs(photo_folder, exist_ok=True)
    os.makedirs(illus_folder, exist_ok=True)
    for folder in (photo_folder, illus_folder):
        for fname in os.listdir(folder):
            fp = os.path.join(folder, fname)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass

    model, tokenizer, image_processor, device = load_clip_model()

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    all_files = []
    for pat in patterns:
        all_files += glob.glob(os.path.join(input_folder, pat))

    for img_path in all_files:
        try:
            is_photo = is_photo_by_labels(
                image_path=img_path,
                model=model,
                tokenizer=tokenizer,
                image_processor=image_processor,
                device=device,
                photo_labels=photo_labels,
                illus_labels=illus_labels
            )
        except Exception as e:
            print(f"WARNING: Failed on {img_path}: {e}")
            is_photo = False

        dest = photo_folder if is_photo else illus_folder
        try:
            img = Image.open(img_path)
            img.save(os.path.join(dest, os.path.basename(img_path)))
        except Exception as e:
            print(f"WARNING: Could not save {img_path} to {dest}: {e}")

    print(f"Filtering done. Photos → '{photo_folder}', Illustrations → '{illus_folder}'.")
