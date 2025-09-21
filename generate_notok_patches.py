# =========================
# Synthetic NOT_OK via CutPaste + Scratch/Line
# =========================
import os, csv, math, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ---------- CONFIG ----------
OK_DIR           = "/kaggle/working/crops/ok"                 # your existing OK patches (224x224)
NOTOK_DIR        =  "/kaggle/working/crops/not_ok"
NOTOK_MANIFEST   = "/kaggle/working/crops/not_ok_manifest.csv"

SYNTH_PER_OK     = 1  # number of synthetic variants to create per OK patch

# Operators to sample from (name, probability)
SYNTH_TYPES_WITH_P = [
    ("cutpaste", 0.6),
    ("scratch",  0.4),
]

# ---- CutPaste params ----
CUT_AREA_FRAC    = (0.10, 0.20)   # ↑ larger pasted piece
ASPECT_RANGE     = (0.4,  2.5)
ROT_DEG          = (-35, 35)      # ↑ more rotation
SCALE_RANGE      = (0.80, 1.25)   # ↑ wider scaling
OFFSET_FRAC      = 1.0/4.0        # ↑ can shift farther
EDGE_FEATHER     = 0.8            # ↓ slightly sharper edges
ALPHA_BLEND_CP   = (0.90, 1.00)   # ↑ more opaque → more visible

# ---- Scratch params ----
SCRATCH_N_LINES      = (2, 5)         # ↑ more lines
SCRATCH_THICKNESS    = (2, 3)         # ↑ thicker lines
SCRATCH_LEN_FRAC     = (0.50, 1.00)   # ↑ longer strokes
SCRATCH_BLUR_SIGMA   = (0.0, 0.6)     # ↓ sharper edges
SCRATCH_INTENSITY    = (0.15, 0.95)   # ↑ allow very dark or very bright
ALPHA_BLEND_SCRATCH  = (0.85, 1.00)   # ↑ more opaque → clearer scratches

# RNG
SEED = 1337

# Assert patch size assumptions
PATCH_W = 224
PATCH_H = 224
# ----------------------------

random.seed(SEED)
np.random.seed(SEED)
Path(NOTOK_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(NOTOK_MANIFEST)).mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# ---------- I/O helpers ----------
def load_rgb01(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb

def save_rgb01(path: str, img01: np.ndarray):
    arr = (np.clip(img01, 0, 1) * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

# ---------- utils ----------
def _rand_uniform(a, b): return a + (b - a) * random.random()

def parent_from_ok_filename(fname: str):
    # e.g. good_000_i1_x120_y45_w224_h224.jpg -> parent "good_000"
    stem = Path(fname).stem
    parts = stem.split("_i")
    return parts[0] if len(parts) > 1 else stem

def list_ok_patches(ok_dir: str):
    paths = [str(p) for p in Path(ok_dir).glob("*") if p.suffix.lower() in IMG_EXTS]
    paths.sort()
    return paths

# ---------- manifest ----------
def init_manifest(csv_path: str):
    write_header = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if write_header:
        w.writerow([
            "patch_path",
            "parent_id",
            "base_ok_filename",
            "synth_type",
            "mask_area_frac",     # cutpaste: actual mask area; scratch: approx drawn area fraction
            "alpha",
            "rot_deg", "scale",   # cutpaste only
            "dx","dy",            # cutpaste only
            "cut_x","cut_y","cut_w","cut_h",  # cutpaste only
            "n_lines","thickness","len_frac","blur_sigma","intensity",  # scratch only
            "rng_seed",
            "label"
        ])
    return f, w

# ---------- CutPaste ----------
def _soft_mask_from_rect(h, w, x, y, cw, ch, feather_sigma=1.0):
    M = np.zeros((h, w), np.uint8)
    x0 = max(0, min(w-1, x))
    y0 = max(0, min(h-1, y))
    x1 = max(0, min(w, x + cw))
    y1 = max(0, min(h, y + ch))
    if x1 <= x0 or y1 <= y0:
        return (M > 0).astype(np.float32)
    M[y0:y1, x0:x1] = 255
    if feather_sigma > 0:
        M = cv2.GaussianBlur(M, (0,0), sigmaX=feather_sigma, sigmaY=feather_sigma)
    return (M.astype(np.float32) / 255.0)

def _affine_on(img, M, out_wh, border_mode=cv2.BORDER_REFLECT):
    h, w = out_wh[1], out_wh[0]
    flags = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    return cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=border_mode)

def op_cutpaste(patch01: np.ndarray):
    H, W = patch01.shape[:2]
    assert (H, W) == (PATCH_H, PATCH_W)
    area = _rand_uniform(*CUT_AREA_FRAC) * (W * H)
    ar = _rand_uniform(*ASPECT_RANGE)  # h/w
    cw = int(round(math.sqrt(area / ar)))
    ch = int(round(cw * ar))
    cw = max(6, min(W//2, cw))
    ch = max(6, min(H//2, ch))
    x = random.randint(0, W - cw)
    y = random.randint(0, H - ch)
    mask = _soft_mask_from_rect(H, W, x, y, cw, ch, EDGE_FEATHER)

    rot = _rand_uniform(*ROT_DEG)
    scl = _rand_uniform(*SCALE_RANGE)
    max_off = int(OFFSET_FRAC * min(W, H))
    dx = random.randint(-max_off, max_off)
    dy = random.randint(-max_off, max_off)

    cx, cy = x + cw/2.0, y + ch/2.0
    M = cv2.getRotationMatrix2D((cx, cy), rot, scl)
    M[:, 2] += [dx, dy]
    moved_rgb = _affine_on(patch01, M, (W, H), border_mode=cv2.BORDER_REFLECT)
    moved_msk = _affine_on(mask,     M, (W, H), border_mode=cv2.BORDER_CONSTANT)
    moved_msk = np.clip(moved_msk, 0.0, 1.0)

    alpha = _rand_uniform(*ALPHA_BLEND_CP)
    m = moved_msk[..., None]
    out01 = patch01 * (1.0 - m) + (alpha * moved_rgb + (1.0 - alpha) * patch01) * m
    out01 = np.clip(out01, 0.0, 1.0)

    params = dict(
        mask_area_frac=float((moved_msk > 0.5).mean()),
        alpha=float(alpha),
        rot_deg=float(rot),
        scale=float(scl),
        dx=int(dx), dy=int(dy),
        cut_x=int(x), cut_y=int(y), cut_w=int(cw), cut_h=int(ch),
        n_lines="", thickness="", len_frac="", blur_sigma="", intensity=""
    )
    return out01, params

# ---------- Scratch/Line ----------
def op_scratch(patch01: np.ndarray):
    H, W = patch01.shape[:2]
    assert (H, W) == (PATCH_H, PATCH_W)
    out = patch01.copy()

    n_lines = random.randint(*SCRATCH_N_LINES)
    thickness = random.randint(*SCRATCH_THICKNESS)
    # approximate area coverage we'll log (rough heuristic)
    approx_area = 0.0

    # draw on an overlay to enable alpha blending
    overlay = out.copy()
    for _ in range(n_lines):
        # pick length
        L = _rand_uniform(*SCRATCH_LEN_FRAC) * min(H, W)
        # random start point & direction
        x0 = random.randint(0, W-1)
        y0 = random.randint(0, H-1)
        angle = _rand_uniform(0, 2*math.pi)
        x1 = int(np.clip(x0 + L * math.cos(angle), 0, W-1))
        y1 = int(np.clip(y0 + L * math.sin(angle), 0, H-1))

        # line intensity (grayscale value), then broadcast to 3 channels
        inten = _rand_uniform(*SCRATCH_INTENSITY)
        color = np.array([inten, inten, inten], dtype=np.float32)

        # draw line on overlay
        p0 = (int(x0), int(y0))
        p1 = (int(x1), int(y1))
        cv2.line(overlay, p0, p1, color.tolist(), thickness=thickness, lineType=cv2.LINE_AA)
        approx_area += (thickness * max(1, int(L))) / (H * W)

    # slight blur to soften the lines
    blur_sigma = _rand_uniform(*SCRATCH_BLUR_SIGMA)
    if blur_sigma > 0:
        overlay = cv2.GaussianBlur(overlay, (0,0), blur_sigma)

    # alpha blend scratches into patch
    alpha = _rand_uniform(*ALPHA_BLEND_SCRATCH)
    out01 = np.clip(alpha * overlay + (1.0 - alpha) * out, 0.0, 1.0)

    params = dict(
        mask_area_frac=float(approx_area),  # heuristic
        alpha=float(alpha),
        rot_deg="", scale="", dx="", dy="", cut_x="", cut_y="", cut_w="", cut_h="",
        n_lines=int(n_lines),
        thickness=int(thickness),
        len_frac=float(_rand_uniform(*SCRATCH_LEN_FRAC)),  # representative
        blur_sigma=float(blur_sigma),
        intensity=float(inten)  # last line's intensity; good enough for logging
    )
    return out01, params

# ---------- operator sampling ----------
def sample_operator():
    names, probs = zip(*SYNTH_TYPES_WITH_P)
    u = random.random()
    s = 0.0
    for name, p in zip(names, probs):
        s += p
        if u <= s:
            return name
    return names[-1]

# ---------- driver ----------
def run_all():
    ok_paths = list_ok_patches(OK_DIR)
    if not ok_paths:
        print(f"[ERR] No OK patches found in {OK_DIR}")
        return

    f, w = init_manifest(NOTOK_MANIFEST)
    total = 0
    try:
        for ok_path in ok_paths:
            ok_img = load_rgb01(ok_path)
            base_name = Path(ok_path).name
            parent_id = parent_from_ok_filename(base_name)

            for v in range(SYNTH_PER_OK):
                # per-variant seed for reproducibility
                rng_seed = (SEED * 10007 + hash((base_name, v))) & 0xffffffff
                random.seed(rng_seed)
                np.random.seed(rng_seed & 0x7fffffff)

                op = sample_operator()
                if op == "cutpaste":
                    out_img, params = op_cutpaste(ok_img)
                elif op == "scratch":
                    out_img, params = op_scratch(ok_img)
                else:
                    raise ValueError(f"Unknown operator: {op}")

                out_name = Path(base_name).stem + f"_synth{op.capitalize()}_v{v+1}.jpg"
                out_path = str(Path(NOTOK_DIR) / out_name)
                save_rgb01(out_path, out_img)

                w.writerow([
                    out_path,
                    parent_id,
                    base_name,
                    op,
                    params["mask_area_frac"],
                    params["alpha"],
                    params["rot_deg"], params["scale"],
                    params["dx"], params["dy"],
                    params["cut_x"], params["cut_y"], params["cut_w"], params["cut_h"],
                    params["n_lines"], params["thickness"], params["len_frac"], params["blur_sigma"], params["intensity"],
                    rng_seed,
                    "not_ok"
                ])
                total += 1
    finally:
        f.close()

    print(f"[DONE] Wrote {total} synthetic NOT_OK patches to {NOTOK_DIR}")
    print(f"[INFO] Manifest: {NOTOK_MANIFEST}")

# ---- run ----
if __name__ == "__main__":
    print("[INFO] Generating synthetic NOT_OK via CutPaste + Scratch…")
    run_all()
