import os
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

META_DIR = ROOT_DIR / "meta"
META_DIR.mkdir(exist_ok=True)

OUT_FEATS = META_DIR / "gallery_dino.npy"
OUT_IDS = META_DIR / "gallery_ids.txt"
OUT_PATHS = META_DIR / "gallery_paths.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vits14" 


def load_model_and_transform():
    print("Root dir:", ROOT_DIR)
    print("Data dir:", DATA_DIR)
    print("Meta dir:", META_DIR)
    print("Using device:", DEVICE)

    print("Loading DINOv2 model...", MODEL_NAME)
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(518, interpolation=Image.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform


@torch.no_grad()
def extract_one_image(img_path: Path, model, transform):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    feats = model(x)                            # (1, D)
    feats = feats.squeeze(0).cpu().numpy().astype("float32")

    # L2 normalize
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return feats


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR không tồn tại: {DATA_DIR}")

    model, transform = load_model_and_transform()

    all_feats = []
    all_ids = []
    all_paths = []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for landmark_dir in sorted(DATA_DIR.iterdir()):
        if not landmark_dir.is_dir():
            continue

        landmark_id = landmark_dir.name
        print(f"\n=== Processing landmark: {landmark_id} ===")

        for img_path in sorted(landmark_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue

            try:
                feat = extract_one_image(img_path, model, transform)
            except Exception as e:
                print(f"  [SKIP] {img_path.name}: {e}")
                continue

            all_feats.append(feat)
            all_ids.append(landmark_id)
            rel_path = img_path.relative_to(ROOT_DIR)
            all_paths.append(str(rel_path))

    if not all_feats:
        print("Không tìm thấy ảnh nào trong DATA_DIR, dừng.")
        return

    all_feats = np.stack(all_feats, axis=0)  # (N, D)
    print("\nTotal images encoded:", all_feats.shape[0])
    print("Feature dim:", all_feats.shape[1])

    np.save(OUT_FEATS, all_feats)
    print("Saved features to", OUT_FEATS)

    with open(OUT_IDS, "w", encoding="utf-8") as f:
        for lid in all_ids:
            f.write(lid + "\n")
    print("Saved landmark ids to", OUT_IDS)

    with open(OUT_PATHS, "w", encoding="utf-8") as f:
        for p in all_paths:
            f.write(p + "\n")
    print("Saved image paths to", OUT_PATHS)


if __name__ == "__main__":
    main()
