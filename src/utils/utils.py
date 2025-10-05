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

@torch.no_grad()
def get_error_map_v1(teacher, student, loader_or_batch):
    """
    Compute anomaly score maps at 64x64 for either:
      - a full DataLoader that yields (paths, images)
      - a single batch (paths, images)

    Returns: np.ndarray of shape [N, 64, 64]
    """
    teacher.eval()
    student.eval()

    # Figure out what we got (full loader or a single batch)
    if hasattr(loader_or_batch, "dataset"):
        iterable = loader_or_batch
        N = len(loader_or_batch.dataset)
    elif isinstance(loader_or_batch, (list, tuple)) and len(loader_or_batch) == 2:
        paths, x = loader_or_batch
        iterable = [loader_or_batch]
        N = x.size(0)
    elif isinstance(loader_or_batch, list) and loader_or_batch and len(loader_or_batch[0]) == 2:
        iterable = loader_or_batch
        N = sum(batch[1].size(0) for batch in loader_or_batch)
    else:
        raise ValueError("get_error_map expects a DataLoader or (paths, images) batch.")

    loss_map = np.zeros((N, 64, 64), dtype=np.float32)
    i = 0

    for paths, batch_img in iterable:      # <<== (paths, images)
        batch_img = batch_img.cuda(non_blocking=True)

        # forward
        t_feat = teacher(batch_img)
        s_feat = student(batch_img)

        # aggregate per-level mismatch into a 64x64 map via product
        score_map = 1.0
        for j in range(len(t_feat)):
            t = F.normalize(t_feat[j], dim=1)
            s = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t - s) ** 2, dim=1, keepdim=True)                   # [B,1,h,w]
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            score_map = score_map * sm                                          # product aggregate

        B = batch_img.size(0)
        loss_map[i:i+B] = score_map.squeeze(1).detach().cpu().numpy()
        i += B

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
def mine_batch(emb, labels, margin):
    """
    emb:    [B, D] tensor of embeddings (batch size B, embedding dim D).
    labels: [B] tensor of class labels (0 = ok, 1 = not_ok).
    margin: triplet margin hyperparameter.

    Returns three lists of indices (a_idx, p_idx, n_idx) so you can index into emb:
        emb[a_idx], emb[p_idx], emb[n_idx]
    """

    cap_percentile = 0.10
    # Cosine similarity matrix: sim[i,j] = cos_sim(emb[i], emb[j])
    sim  = emb @ emb.t()   # [B,B]

    # Convert similarity into distance:
    # cos_dist = 1 - cos_sim (0 = identical, 2 = opposite).
    dist = 1.0 - sim       # [B,B]

    # Lists to hold the triplet indices
    a_idx, p_idx, n_idx = [], [], []

    # ---- STEP 1: Loop over every sample as anchor ----
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
        pj = pos[torch.argmax(dist[i, pos])]
        # pj = pos[torch.argmin(dist[i, pos])]  

        # ---- STEP 4: Choose the negative ----
         # ------- semi-hard negative: d(a,n) in (d(a,p), d(a,p)+margin) -------
        neg_ids   = torch.where(diff)[0]
        neg_dists = dist[i, diff]
        band = (neg_dists > dist[i, pj]) & (neg_dists < dist[i, pj] + margin)

        if band.any():
            # choose the closest semi-hard negative (tightest violation)
            nj = neg_ids[torch.argmin(neg_dists[band])]
        else:
            # fallback: avoid pathologically hard outliers
            # keep only closest k% negatives, then take argmin
            if neg_dists.numel() > 0:
                k = max(1, int(cap_percentile * neg_dists.numel()))
                topk_vals, topk_idx = torch.topk(-neg_dists, k=k)  # largest(-d) == smallest(d)
                nj = neg_ids[topk_idx[-1]]                          # farthest among the "closest k%" -> moderate hard
            else:
                continue

        # ---- STEP 5: Store the triplet indices ----
        a_idx.append(i)
        p_idx.append(pj.item())
        n_idx.append(nj.item())

    return a_idx, p_idx, n_idx


def xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

def merge_touching_boxes_xywh(boxes, gap=0, iou_thr=0.0, max_iters=5):
    """
    Merge boxes that intersect or are within `gap` pixels of touching.
    boxes: list[(x,y,w,h)]  in image coordinates
    gap:   extra tolerance (in pixels). Example: 2–6 px, or int(0.01*min(H,W))
    iou_thr: also merge if IoU >= iou_thr (e.g., 0.0 merges any intersection)
    max_iters: do multiple passes until stable or limit hit
    """
    if not boxes:
        return boxes

    # work in xyxy
    cur = [xywh_to_xyxy(b) for b in boxes]

    def _union(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))

    changed = True
    it = 0
    while changed and it < max_iters:
        it += 1
        changed = False
        cur.sort(key=lambda b: (b[0], b[1], b[2], b[3]))  # mild determinism
        merged = []
        used = [False] * len(cur)

        for i in range(len(cur)):
            if used[i]:
                continue
            a = cur[i]
            ax1, ay1, ax2, ay2 = a
            group = a
            used[i] = True

            # try to absorb any overlapping / touching neighbors
            j = i + 1
            while j < len(cur):
                if used[j]:
                    j += 1
                    continue
                b = cur[j]
                # quick reject by x projection when far apart to the right (speeds up)
                if b[0] > group[2] + gap and iou(group, b) < iou_thr:
                    # since sorted by x1, boxes ahead start even farther right
                    j += 1
                    continue

                # merge condition: touch/near OR IoU ≥ threshold
                if boxes_touch_or_near(group, b, gap=gap) or iou(group, b) >= iou_thr:
                    group = _union(group, b)
                    used[j] = True
                    changed = True
                j += 1

            merged.append(group)

        cur = merged

    # back to xywh
    return [xyxy_to_xywh(b) for b in cur]

import numpy as np

def _xywh_to_xyxy(boxes):
    """
    boxes: (N,4) [x,y,w,h]  -> (N,4) [x1,y1,x2,y2]
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0,4)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2] - 1.0
    y2 = boxes[:,1] + boxes[:,3] - 1.0
    return np.stack([x1,y1,x2,y2], axis=1)

def _iou_matrix(boxes_a, boxes_b):
    """
    boxes_a: (Na,4) xyxy, boxes_b: (Nb,4) xyxy
    returns IoU matrix (Na, Nb)
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    A = boxes_a.astype(np.float32)
    B = boxes_b.astype(np.float32)

    area_a = (np.clip(A[:,2] - A[:,0] + 1, 0, None) *
              np.clip(A[:,3] - A[:,1] + 1, 0, None))  # (Na,)
    area_b = (np.clip(B[:,2] - B[:,0] + 1, 0, None) *
              np.clip(B[:,3] - B[:,1] + 1, 0, None))  # (Nb,)

    x1 = np.maximum(A[:, None, 0], B[None, :, 0])
    y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2])
    y2 = np.minimum(A[:, None, 3], B[None, :, 3])

    inter_w = np.clip(x2 - x1 + 1, 0, None)
    inter_h = np.clip(y2 - y1 + 1, 0, None)
    inter = inter_w * inter_h  # (Na,Nb)

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou

def _greedy_match(iou_mat, thr=0.5):
    """
    Greedy one-to-one matching by IoU.
    Returns:
      matches: list of (pred_idx, gt_idx, iou)
      unmatched_pred: list of pred indices
      unmatched_gt:   list of gt  indices
    """
    Na, Nb = iou_mat.shape
    used_pred = np.zeros(Na, dtype=bool)
    used_gt   = np.zeros(Nb, dtype=bool)

    # process pairs in descending IoU
    flat = iou_mat.reshape(-1)
    order = np.argsort(-flat)  # descending
    matches = []
    for f in order:
        pi = f // Nb
        gi = f %  Nb
        if used_pred[pi] or used_gt[gi]:
            continue
        iou = iou_mat[pi, gi]
        if iou < thr:
            break
        used_pred[pi] = True
        used_gt[gi]   = True
        matches.append((pi, gi, float(iou)))

    unmatched_pred = np.where(~used_pred)[0].tolist()
    unmatched_gt   = np.where(~used_gt)[0].tolist()
    return matches, unmatched_pred, unmatched_gt

def compute_iou_localization(pred_by_img, gt_by_img, iou_thr=0.5):
    """
    Computes localization stats at IoU threshold (e.g., 0.5):
      - mean IoU over matched TPs
      - TP / FP / FN counts
      - also returns list of IoUs for all TPs
    """
    iou_tps = []
    TP = FP = FN = 0

    for img_path, pred in pred_by_img.items():
        p_boxes = pred.get("boxes", [])
        # sort by score (high→low) so matching behavior is deterministic
        p_scores = pred.get("scores", [])
        if len(p_boxes) != len(p_scores):
            # fallback: no scores provided
            order = np.arange(len(p_boxes))
        else:
            order = np.argsort(-np.asarray(p_scores, dtype=np.float32))
        p_boxes = [p_boxes[i] for i in order]

        g = gt_by_img.get(img_path, {"boxes": []})
        g_boxes = g.get("boxes", [])

        P = _xywh_to_xyxy(np.asarray(p_boxes, dtype=np.float32))
        G = _xywh_to_xyxy(np.asarray(g_boxes, dtype=np.float32))

        iou_mat = _iou_matrix(P, G)
        matches, un_p, un_g = _greedy_match(iou_mat, thr=iou_thr)

        TP += len(matches)
        FP += len(un_p)
        FN += len(un_g)

        for _, _, iou in matches:
            iou_tps.append(iou)

    mean_iou_tp = float(np.mean(iou_tps)) if iou_tps else 0.0
    return mean_iou_tp, iou_tps, TP, FP, FN
