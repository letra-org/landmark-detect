# app_gradio.py
import json
import io
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import gradio as gr

# ================== ĐƯỜNG DẪN PROJECT ==================
# scripts/
ROOT_DIR = Path(__file__).resolve().parents[1]
META_DIR = ROOT_DIR / "meta"

FEATS_PATH = META_DIR / "gallery_dino.npy"
IDS_PATH = META_DIR / "gallery_ids.txt"
CATALOG_JSON = META_DIR / "catalog.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vits14"


# ================== LOAD MODEL & TRANSFORM =============
def load_model_and_transform():
    print("[INFO] Using device:", DEVICE)
    print("[INFO] Loading DINOv2 model for Gradio...", MODEL_NAME)

    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME).to(DEVICE)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(518, interpolation=Image.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


@torch.no_grad()
def encode_image(img: Image.Image, model, transform) -> np.ndarray:
    img = img.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    feats = model(x)  # (1, D)
    feats = feats.squeeze(0).cpu().numpy().astype("float32")
    # L2 normalize
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return feats  # (D,)


# ================== LOAD GALLERY & CATALOG =============
def load_gallery_and_catalog():
    print("[INFO] Loading gallery features...")
    gallery_feats = np.load(FEATS_PATH)
    print("  -> gallery_feats shape:", gallery_feats.shape)

    print("[INFO] Loading gallery IDs...")
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        gallery_ids = [line.strip() for line in f if line.strip()]

    if len(gallery_ids) != gallery_feats.shape[0]:
        raise ValueError(
            f"Số dòng gallery_ids ({len(gallery_ids)}) "
            f"không khớp số vector ({gallery_feats.shape[0]})"
        )

    print("[INFO] Loading catalog.json...")
    with open(CATALOG_JSON, "r", encoding="utf-8") as f:
        catalog_list = json.load(f)

    id2meta: Dict[str, Dict[str, Any]] = {item["id"]: item for item in catalog_list}
    print("[INFO] Catalog entries:", len(id2meta))

    return gallery_feats, gallery_ids, id2meta


# ================== HÀM XỬ LÝ CHÍNH ====================
def format_top1_info(meta: Dict[str, Any], score: float) -> str:
    """Trả về chuỗi markdown mô tả chi tiết cho kết quả tốt nhất."""
    lines = []
    lines.append(f"### ✅ Kết quả nhận diện")
    lines.append(f"**ID**: `{meta.get('id', '')}`")
    lines.append(f"**Tên (VI)**: {meta.get('name_vi', meta.get('id', ''))}")
    name_en = meta.get("name_en")
    if name_en:
        lines.append(f"**Tên (EN)**: {name_en}")
    lines.append(f"**Similarity**: `{score:.4f}`")

    short_intro = meta.get("short_intro")
    if short_intro:
        lines.append("")
        lines.append(f"**Giới thiệu ngắn:** {short_intro}")

    fun_facts = meta.get("fun_facts") or []
    if fun_facts:
        lines.append("")
        lines.append("**Fun facts:**")
        for ff in fun_facts[:3]:
            lines.append(f"- {ff}")

    story = meta.get("story")
    if story:
        lines.append("")
        lines.append("**Câu chuyện / truyền thuyết:**")
        lines.append(story)

    highlights = meta.get("highlights") or []
    if highlights:
        lines.append("")
        lines.append("**Điểm nhấn nổi bật:** " + ", ".join(highlights))

    location = meta.get("location")
    if location:
        lines.append("")
        lines.append(f"**Địa điểm:** {location}")

    wiki_url = meta.get("wiki_url")
    if wiki_url:
        lines.append("")
        lines.append(f"[🔗 Xem thêm trên Wikipedia]({wiki_url})")

    return "\n".join(lines)


def predict(
    img: Image.Image,
    top_k: int,
    model,
    transform,
    gallery_feats,
    gallery_ids,
    id2meta,
):
    if img is None:
        return "Vui lòng upload một bức ảnh danh lam thắng cảnh.", None

    # Encode query
    q = encode_image(img, model, transform)  # (D,)
    sims = gallery_feats @ q  # (N,)
    idxs = np.argsort(-sims)[:top_k]

    results_table: List[Dict[str, Any]] = []
    for rank, idx in enumerate(idxs, start=1):
        lid = gallery_ids[idx]
        score = float(sims[idx])
        meta = id2meta.get(lid, {"id": lid})
        results_table.append(
            {
                "Rank": rank,
                "ID": lid,
                "Tên (VI)": meta.get("name_vi", lid),
                "Tên (EN)": meta.get("name_en", ""),
                "Similarity": round(score, 4),
            }
        )

    # Lấy best result cho phần mô tả chi tiết
    best_idx = idxs[0]
    best_id = gallery_ids[best_idx]
    best_score = float(sims[best_idx])
    best_meta = id2meta.get(best_id, {"id": best_id})

    detail_markdown = format_top1_info(best_meta, best_score)
    return detail_markdown, results_table


# ================== MAIN (GRADIO) ======================
if __name__ == "__main__":
    # Load model & data một lần
    model, transform = load_model_and_transform()
    gallery_feats, gallery_ids, id2meta = load_gallery_and_catalog()

    def gradio_predict(img, top_k):
        return predict(
            img,
            top_k,
            model,
            transform,
            gallery_feats,
            gallery_ids,
            id2meta,
        )

    demo = gr.Interface(
        fn=gradio_predict,
        inputs=[
            gr.Image(type="pil", label="Upload ảnh danh lam thắng cảnh"),
            gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Top-K kết quả muốn xem",
            ),
        ],
        outputs=[
            gr.Markdown(label="Thông tin chi tiết kết quả Top-1"),
            gr.Dataframe(
                headers=["Rank", "ID", "Tên (VI)", "Tên (EN)", "Similarity"],
                label="Top-K kết quả",
            ),
        ],
        title="Landmark VN Retrieval – DINOv2 (Gradio Demo)",
        description=(
            "Upload một bức ảnh danh lam thắng cảnh Việt Nam, hệ thống sẽ truy hồi "
            "Top-K địa danh giống nhất bằng DINOv2 + cosine similarity."
        ),
        # Gradio v4 dùng flagging_mode, không dùng allow_flagging nữa
        flagging_mode="never",
    )

    demo.launch()
