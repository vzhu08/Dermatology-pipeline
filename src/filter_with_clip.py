# src/filter_with_clip.py

import os
import glob
import math
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed


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
) -> dict:
    """
    Compute cosine similarities between a single image and two sets of text labels,
    then classify as 'photo' or 'illus' based on mean scores.

    Returns a dict:
      {
        "photo_scores": { label: score, ... },
        "illus_scores": { label: score, ... },
        "mean_photo": float,
        "mean_illus": float,
        "classification": "photo" or "illus"
      }
    """
    # Load model and labels
    model, tokenizer, image_processor, device = load_clip_model()
    all_labels = photo_labels + illus_labels

    # Embed labels
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Load and embed the image
    img = Image.open(image_path).convert("RGB")
    img_inputs = image_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feat = model.get_image_features(pixel_values=img_inputs["pixel_values"])
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        sims = (img_feat @ text_embeds.T).squeeze(0).cpu().numpy()

    # Split scores
    photo_scores = {photo_labels[i]: float(sims[i]) for i in range(len(photo_labels))}
    illus_scores = {illus_labels[j]: float(sims[len(photo_labels) + j])
                    for j in range(len(illus_labels))}

    # Compute means and classification
    mean_photo = np.mean(list(photo_scores.values())) if photo_scores else float("-inf")
    mean_illus = np.mean(list(illus_scores.values())) if illus_scores else float("-inf")
    classification = "photo" if mean_photo > mean_illus else "illus"

    return {
        "photo_scores": photo_scores,
        "illus_scores": illus_scores,
        "mean_photo": mean_photo,
        "mean_illus": mean_illus,
        "classification": classification
    }


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
    img_inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

    all_labels = photo_labels + illus_labels
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=img_inputs["pixel_values"]
        )
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
        sims = (img_embeds @ text_embeds.T).squeeze(0).cpu().numpy()

    num_photo = len(photo_labels)
    max_photo = float(np.max(sims[:num_photo])) if num_photo > 0 else float("-inf")
    max_illus = float(np.max(sims[num_photo:])) if illus_labels else float("-inf")

    return max_photo > max_illus


def _process_batch_thread(
    batch_paths: list[str],
    text_embeds: torch.Tensor,
    all_labels: list[str],
    photo_labels: list[str],
    output_folder: str,
    model: CLIPModel,
    image_processor: CLIPImageProcessor,
    device: str
):
    """
    Worker thread: embed a batch of images, determine class by mean score,
    and save into output_folder/photo/ or output_folder/illus/.
    """
    imgs, paths = [], []
    for p in batch_paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            paths.append(p)
        except:
            continue
    if not imgs:
        return

    inputs = image_processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        sims = (feats @ text_embeds.T).cpu().numpy()

    n_photo = len(photo_labels)
    for i, img_path in enumerate(paths):
        scores = sims[i]
        mean_photo = scores[:n_photo].mean()
        mean_illus = scores[n_photo:].mean()
        cls = "photo" if mean_photo > mean_illus else "illus"
        dest_dir = os.path.join(output_folder, cls)
        try:
            Image.open(img_path).save(os.path.join(dest_dir, os.path.basename(img_path)))
        except:
            pass


def filter_with_clip(
    input_folder: str,
    output_folder: str,
    photo_labels: list[str],
    illus_labels: list[str],
    batch_size: int = 32,
    max_workers: int = None
):
    """
    Walk through all images in input_folder and classify each into
    output_folder/photo/ or output_folder/illus/ based on mean CLIP scores,
    processing batches in parallel threads.
    """
    # Prepare output folders
    for cls in ("photo", "illus"):
        cls_path = os.path.join(output_folder, cls)
        if os.path.isdir(cls_path):
            for fname in os.listdir(cls_path):
                try:
                    os.remove(os.path.join(cls_path, fname))
                except OSError:
                    pass
        else:
            os.makedirs(cls_path, exist_ok=True)

    # Load model + processor
    model, tokenizer, image_processor, device = load_clip_model()
    all_labels = photo_labels + illus_labels

    # Embed all labels once
    text_inputs = tokenizer(all_labels, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Gather files and create batches
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(input_folder, pat))
    files.sort()
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    # Process in parallel threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            futures.append(executor.submit(
                _process_batch_thread,
                batch, text_embeds, all_labels,
                photo_labels, output_folder,
                model, image_processor, device
            ))
        for fut in as_completed(futures):
            fut.result()

    print(f"Done. Sorted into '{output_folder}/photo/' and '{output_folder}/illus/'.")
