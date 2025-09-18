# from roboflow import Roboflow

# rf = Roboflow(api_key="VUR5Jn1GLOzjx7FNbmUY")
# ds = rf.workspace("ts-nnhge").project("defect-labeling-ffxog").version(1).download("folder")

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np

# ==== CONFIG ====
SRC_OK_DIR = "dataset_raw/ok"          # your source OK images
DST_OK_DIR = "dataset_train/train/ok"  # where new crops will be saved (triplet training)
FINAL_SIZE = 224
CROPS_PER_IMAGE = 6
VARIANCE_MIN = 12.0    # reject too-flat crops; lower -> keep more, higher -> keep fewer
SEED = 42              # for reproducibility
# =================

random.seed(SEED)
np.random.seed(SEED)
Path(DST_OK_DIR).mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def ensure_min_side(img: Image.Image, min_side: int) -> Image.Image:
    """Upscale (if needed) so min(H, W) >= min_side; keeps aspect ratio."""
    w, h = img.size
    smin = min(w, h)
    if smin >= min_side:
        return img
    scale = min_side / smin
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)

def random_square_crop(img: Image.Image, side: int) -> Image.Image:
    """Random 224x224 crop without stretching; assumes min side >= side."""
    w, h = img.size
    if w == side and h == side:
        return img
    x0 = random.randint(0, w - side)
    y0 = random.randint(0, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side))

def variance_ok(pil_img: Image.Image, thresh: float) -> bool:
    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    return float(arr.var()) >= thresh

def process_one(img_path: Path):
    try:
        img = load_rgb(img_path)
    except Exception as e:
        print(f"[skip] Cannot open {img_path}: {e}")
        return

    img = ensure_min_side(img, FINAL_SIZE)

    made, tries = 0, 0
    max_tries = CROPS_PER_IMAGE * 10  # plenty of chances to pass variance
    while made < CROPS_PER_IMAGE and tries < max_tries:
        tries += 1
        crop = random_square_crop(img, FINAL_SIZE)
        if not variance_ok(crop, VARIANCE_MIN):
            continue
        out_name = f"{img_path.stem}_crop{made+1}.jpg"
        out_path = Path(DST_OK_DIR) / out_name
        crop.save(out_path, quality=95)
        made += 1

    if made < CROPS_PER_IMAGE:
        print(f"[warn] {img_path.name}: created {made}/{CROPS_PER_IMAGE} crops (variance filter may be strict)")

def main():
    srcs = [p for p in Path(SRC_OK_DIR).glob("**/*") if p.suffix.lower() in IMG_EXTS]
    if not srcs:
        print(f"[err] No images found under {SRC_OK_DIR}")
        return

    print(f"[info] Found {len(srcs)} OK images. Writing crops to: {DST_OK_DIR}")
    for p in srcs:
        process_one(p)
    print("[done] OK random crops generated.")

if __name__ == "__main__":
    main()
