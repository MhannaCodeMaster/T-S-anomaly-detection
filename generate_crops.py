import os
import csv
import random
from pathlib import Path
from PIL import Image
import numpy as np

# ==== CONFIG ====
SRC_OK_DIR = "/kaggle/input/mvtec-ad/cable/train/good"      # source OK images
DST_OK_DIR = "/kaggle/working/train/ok"                     # where crops are saved
MANIFEST_CSV = "/kaggle/working/train/ok_manifest.csv"      # metadata for all crops
FINAL_SIZE = 224
CROPS_PER_IMAGE = 6
VARIANCE_MIN = 12.0   # reject too-flat crops; lower -> keep more, higher -> keep fewer
SEED = 42
# =================

random.seed(SEED)
np.random.seed(SEED)
Path(DST_OK_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(MANIFEST_CSV)).mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def ensure_min_side(img: Image.Image, min_side: int):
    """
    Upscale (if needed) so min(H, W) >= min_side; keeps aspect ratio.
    Returns (resized_img, orig_w, orig_h, new_w, new_h, scale)
    """
    w, h = img.size
    smin = min(w, h)
    if smin >= min_side:
        return img, w, h, w, h, 1.0
    scale = min_side / smin
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC), w, h, new_w, new_h, scale

def random_square_crop(img: Image.Image, side: int):
    """
    Random side x side crop without stretching; assumes min side >= side.
    Returns (cropped_image, x, y, w, h) in the *resized image* coordinate space.
    """
    w, h = img.size
    if w == side and h == side:
        return img, 0, 0, side, side
    x0 = random.randint(0, w - side)
    y0 = random.randint(0, h - side)
    return img.crop((x0, y0, x0 + side, y0 + side)), x0, y0, side, side

def variance_ok(pil_img: Image.Image, thresh: float) -> bool:
    arr = np.asarray(pil_img.convert("L"), dtype=np.float32)
    return float(arr.var()) >= thresh

def init_manifest(csv_path: str):
    write_header = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "patch_path",
            "parent_id",
            "crop_index",
            "x","y","w","h",                 # coords in resized image space
            "orig_w","orig_h",
            "resized_w","resized_h",
            "scale",                         # resized / original
            "label"                          # ok / not_ok (here: ok)
        ])
    return f, writer

def process_one(img_path: Path, writer):
    try:
        img = load_rgb(img_path)
    except Exception as e:
        print(f"[skip] Cannot open {img_path}: {e}")
        return

    img_resized, ow, oh, rw, rh, scale = ensure_min_side(img, FINAL_SIZE)

    made, tries = 0, 0
    max_tries = CROPS_PER_IMAGE * 10  # plenty of chances to pass variance
    parent_id = img_path.stem  # e.g., good_000
    while made < CROPS_PER_IMAGE and tries < max_tries:
        tries += 1
        crop, x, y, w, h = random_square_crop(img_resized, FINAL_SIZE)
        if not variance_ok(crop, VARIANCE_MIN):
            continue

        # Include parent and coords in filename
        out_name = f"{parent_id}_i{made+1}_x{x}_y{y}_w{w}_h{h}.jpg"
        out_path = Path(DST_OK_DIR) / out_name
        crop.save(out_path, quality=95)

        # Write manifest row
        writer.writerow([
            str(out_path),
            parent_id,
            made + 1,
            x, y, w, h,
            ow, oh,
            rw, rh,
            f"{scale:.6f}",
            "ok"
        ])

        made += 1

    if made < CROPS_PER_IMAGE:
        print(f"[warn] {img_path.name}: created {made}/{CROPS_PER_IMAGE} crops (variance filter may be strict)")

def main():
    srcs = [p for p in Path(SRC_OK_DIR).glob("**/*") if p.suffix.lower() in IMG_EXTS]
    if not srcs:
        print(f"[err] No images found under {SRC_OK_DIR}")
        return

    print(f"[info] Found {len(srcs)} OK images. Writing crops to: {DST_OK_DIR}")
    mf, writer = init_manifest(MANIFEST_CSV)
    try:
        for p in srcs:
            process_one(p, writer)
    finally:
        mf.close()
    print("[done] OK random crops generated and manifest written.")

if __name__ == "__main__":
    main()
