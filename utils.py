import torch
import torch.optim
import torch.nn.functional as F
import copy
import os
import cv2
import numpy as np
from pathlib import Path


def get_error_map(teacher, student, loader):
    """Testing function to compute anomaly score maps."""
    teacher.eval()
    student.eval()
    # Pre-allocate array to store the anomaly score map per image (64x64 resolution)
    loss_map = np.zeros((len(loader.dataset), 64, 64))
    i = 0 # Tracks where to write each batch of results in the pre-allocated array.
    
    # Iterate over the data loader
    for batch_data in loader:
        _, batch_img = batch_data # Each batch_data is a tuple (image_path, image_tensor)
        batch_img = batch_img.cuda() # Moving the image tensor to GPU

        # Foward pass      
        with torch.no_grad():
            t_feat = teacher(batch_img)
            s_feat = student(batch_img)
        score_map = 1.
        
        # Per-level mismatch -> upsample -> aggregate.
        for j in range(len(t_feat)):
            # Normalize the feature maps along the channel dimension -> removed scale effects, comapres direction of features.
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            # Compute per-pixel squared L2 distance across channels -> a dense anomaly map for that layer.
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            # Interpolate the score map to a common resolution (64x64)
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            # aggregate score map by element-wise product
            score_map = score_map * sm
        
        # Store batch results into the big buffer.
        # Converts the final score map to NumPy and writes it to the correct slice of loss_map. 
        loss_map[i: i + batch_img.size(0)] = score_map.squeeze().cpu().data.numpy()
        # Advances the index by the batch size.
        i += batch_img.size(0)
    
    # Returns an (N, 64, 64) array of anomaly score maps, where N is the number of images (Higher = more anomalous).
    return loss_map
   
def normalize_01(x: np.ndarray):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)
    
def upscale_heatmap_to_image(hm64: np.ndarray, target_hw):
    H, W = target_hw
    hm_up = cv2.resize(hm64.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    hm_up = normalize_01(hm_up)
    return hm_up

def threshold_heatmap(hm_up, method="percentile", percentile=99.5):
    """
    hm_up: (H,W) float in [0,1]
    Returns: (mask_uint8, threshold_value)
    """
    if method == "otsu":
        hm8 = (hm_up * 255).astype(np.uint8)
        thr_val, mask = cv2.threshold(hm8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = thr_val / 255.0
    elif method == "percentile":
        thr = np.percentile(hm_up, percentile)
        mask = (hm_up >= thr).astype(np.uint8) * 255
    else:  # fixed threshold
        thr = float(method)
        mask = (hm_up >= thr).astype(np.uint8) * 255
    return mask, thr

def components_to_bboxes(mask: np.ndarray, min_area, ignore_border: bool = True):
    """
    mask: uint8 {0,255}, shape (H,W)
    Returns list of boxes: [(x,y,w,h), ...]
    """
    H, W = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for lbl in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def draw_boxes(img_bgr: np.ndarray, boxes, color=(0,255,0), thickness=2):
    vis = img_bgr.copy()
    for (x,y,w,h) in boxes:
        cv2.rectangle(vis, (x,y), (x+w, y+h), color, thickness)
    return vis

def pad_box(x, y, w, h, H, W, pad_ratio=0.05):
    """
    Adds padding to a bounding box, so the crop extends slightly beyond the original defect region.
    """
    px, py = int(w * pad_ratio), int(h * pad_ratio)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    return x0, y0, x1, y1

def is_contained(small_box, large_box, tol=0.9):
    # small_box, large_box: (x,y,w,h)
    xi, yi, wi, hi = small_box
    xo, yo, wo, ho = large_box
    xi1 = xi+wi
    yi1 = yi+hi
    xo1 = xo+wo
    yo1 = yo+ho
    inter_x0 = max(xi, xo)
    inter_y0 = max(yi, yo)
    inter_x1 = min(xi1, xo1)
    inter_y1 = min(yi1, yo1)
    inter_area = max(0, inter_x1-inter_x0) * max(0, inter_y1-inter_y0)
    return inter_area >= tol*(wi*hi)

def remove_nested_boxes(boxes, tolerance=0.9):   
    """Remove boxes that are fully or mostly contained within another box."""
    # first sort boxes by area
    boxes = sorted(boxes, key=lambda b: b[2]*b[3])
    keep = []
    # loop through boxes, smallest to largest
    for i, box in enumerate(boxes):
        drop = False
        # Compare smallest box to all larger boxes
        for j in range(i+1, len(boxes)):
            # if box[i] is contained at least in one of the larger boxes, drop it
            if is_contained(boxes[i], boxes[j], tol=tolerance):
                drop = True
                break
        if not drop:
            keep.append(box)
        
    return keep

def boxes_touch_or_near(a, b, gap=0):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        # Expand by 'gap' and test intersection
        ax1g, ay1g, ax2g, ay2g = ax1-gap, ay1-gap, ax2+gap, ay2+gap
        bx1g, by1g, bx2g, by2g = bx1-gap, by1-gap, bx2+gap, by2+gap
        ix1, iy1 = max(ax1g, bx1g), max(ay1g, by1g)
        ix2, iy2 = min(ax2g, bx2g), min(ay2g, by2g)
        return (ix2 > ix1) and (iy2 > iy1)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9
    return inter / ua

def expand_boxes(boxes, H, W, expand_ratio=0.12):
    """Expand each (x,y,w,h) by a ratio; clip to image size."""
    if expand_ratio <= 0 or not boxes:
        return boxes
    out = []
    for (x, y, w, h) in boxes:
        cx, cy = x + w/2.0, y + h/2.0
        w2 = int(round(w * (1 + expand_ratio)))
        h2 = int(round(h * (1 + expand_ratio)))
        x2 = int(round(cx - w2 / 2.0))
        y2 = int(round(cy - h2 / 2.0))
        x2 = max(0, x2); y2 = max(0, y2)
        w2 = min(W - x2, w2); h2 = min(H - y2, h2)
        out.append((x2, y2, w2, h2))
    return out

def keep_largest_box(boxes):
    """Keep a single largest-area box (optional alternative policy)."""
    if not boxes:
        return boxes
    areas = [w*h for (x,y,w,h) in boxes]
    i = int(np.argmax(areas))
    return [boxes[i]]

def load_calibration(npz_path):
    """
    Loads calibration stats from npz file.
    """
    data = np.load(npz_path)
    return float(data["mean"]), float(data["std"])

def zscore_calibrate(hm_up, mean, std, eps=1e-6):
    return (hm_up - mean) / max(std, eps)

def cosine_border_taper(H, W, margin=6):
    """2D cosine ramp that downweights borders by ~0 at edges and ~1 in center."""
    if margin <= 0:
        return np.ones((H, W), dtype=np.float32)
    y = np.ones(H, np.float32); x = np.ones(W, np.float32)

    wy = ramp(H, margin)
    wx = ramp(W, margin)
    return np.outer(wy, wx)    # (H,W)

def ramp(n, m):
        v = np.ones(n, np.float32)
        if m > 0:
            t = np.linspace(0, np.pi, m, dtype=np.float32)
            v[:m]  = (1 - np.cos(t)) * 0.5
            v[-m:] = v[:m][::-1]
            v[m:-m] = 1.0
        return v

def get_box_scores(boxes, hm, mode="mean"):
    """
    boxes: list of (x,y,w,h)
    hm: calibrated heatmap (float32, HxW)
    mode: "mean" or "max"
    returns: list of scores
    """
    scores = []
    H, W = hm.shape
    for (x, y, w, h) in boxes:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        patch = hm[y0:y1, x0:x1]
        if patch.size == 0:
            scores.append(0.0)
            continue
        if mode == "max":
            scores.append(float(np.max(patch)))
        else:  # mean
            scores.append(float(np.mean(patch)))
    return scores

@torch.no_grad()
def compute_train_calibration_stats(teacher, student, train_loader, cfg, out, device="cuda"):
    """
    Compute scalar μ, σ over all pixels of anomaly maps on normal training images.
    Saves to npz: {mean: float, std: float}.
    Returns mean and std
    """
    print("Computing training set calibration stats...")
    teacher.eval(); student.eval()
    sums, sums2, count = 0.0, 0.0, 0

    for _, batch_img in train_loader:
        batch_img = batch_img.to(device, non_blocking=True)
        t_feat = teacher(batch_img)
        s_feat = student(batch_img)

        # feature mismatch -> (N,1,h,w) per level -> upsample to a common 64x64 -> sum across levels
        score = 0
        for t, s in zip(t_feat, s_feat):
            # per-pixel channel MSE
            diff = (t - s).pow(2).mean(dim=1, keepdim=True)   # (N,1,h,w)
            diff64 = F.interpolate(diff, size=(64, 64), mode="bilinear", align_corners=False)
            score += diff64                                   # (N,1,64,64)

        score_np = score.squeeze(1).float().cpu().numpy()     # (N,64,64)
        sums  += score_np.sum()
        sums2 += (score_np ** 2).sum()
        count += score_np.size

    mean = sums / max(1, count)
    var  = (sums2 / max(1, count)) - (mean * mean)
    std  = float(np.sqrt(max(var, 1e-12)))

    file_path = os.path.join(out["calibration"],'calib_stats.npz')
    np.savez(file_path, mean=float(mean), std=float(std))
    print(f"[calib] saved μ={mean:.6g}, σ={std:.6g}")
    return mean, std


@torch.no_grad()
def mine_batch_hard(emb, labels, margin):
    """
    emb:    [B, D] tensor of embeddings (batch size B, embedding dim D).
    labels: [B] tensor of class labels (0 = ok, 1 = not_ok).
    margin: triplet margin hyperparameter.

    Returns three lists of indices (a_idx, p_idx, n_idx) so you can index into emb:
        emb[a_idx], emb[p_idx], emb[n_idx]
    """

    # ---- STEP 1: Normalize embeddings ----
    # Cosine similarity only makes sense if vectors are normalized.
    emb = F.normalize(emb, p=2, dim=1)

    # Cosine similarity matrix: sim[i,j] = cos_sim(emb[i], emb[j])
    sim  = emb @ emb.t()   # [B,B]

    # Convert similarity into distance:
    # cos_dist = 1 - cos_sim (0 = identical, 2 = opposite).
    dist = 1.0 - sim       # [B,B]

    # Lists to hold the triplet indices
    a_idx, p_idx, n_idx = [], [], []

    # ---- STEP 2: Loop over every sample as anchor ----
    for i in range(len(emb)):
        # Identify same-class samples (positives) and different-class samples (negatives)
        same = (labels == labels[i])    # Boolean mask of positives
        diff = ~same                    # Boolean mask of negatives
        same[i] = False                 # Exclude the anchor itself from positives

        pos = torch.where(same)[0]      # indices of positives
        neg = torch.where(diff)[0]      # indices of negatives
        if len(pos) == 0 or len(neg) == 0:
            continue  # skip if we can't form a triplet

        # ---- STEP 3: Choose the positive ----
        # pj = pos[torch.argmax(dist[i, pos])]
        pj = pos[torch.argmin(dist[i, pos])]  

        # ---- STEP 4: Choose the negative ----
        # Prefer "semi-hard" negatives if possible:
        # - They are further away than the positive (harder),
        # - But not *too* far (still within the margin band).
        band = (dist[i, neg] > dist[i, pj]) & (dist[i, neg] < dist[i, pj] + margin)

        # if torch.any(band):
        #     # If we found semi-hard negatives, pick one randomly
        #     cand = neg[band]
        #     nj   = cand[torch.randint(len(cand), (1,)).item()]
        # else:
        #     # Otherwise, fallback: pick the *hardest* negative,
        #     # i.e. the one closest to the anchor (minimum distance).
        #     nj = neg[torch.argmin(dist[i, neg])]
        nj = neg[torch.argmin(dist[i, neg])]  # batch-hard negative

        # ---- STEP 5: Store the triplet indices ----
        a_idx.append(i)
        p_idx.append(pj.item())
        n_idx.append(nj.item())

    return a_idx, p_idx, n_idx
