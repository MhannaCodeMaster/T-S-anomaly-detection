import torch
import torch.optim
import torch.nn.functional as F
import copy
import os
import cv2
import numpy as np
from pathlib import Path
from hydra.core.hydra_config import HydraConfig


def get_error_map(teacher, student, loader):
    """Testing function to compute anomaly score maps."""
    print("Computing anomaly score maps...")
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
    
    print("anomaly score maps computed.")    
    # Returns an (N, 64, 64) array of anomaly score maps, where N is the number of images (Higher = more anomalous).
    return loss_map

def train_val_student(teacher, student, train_loader, val_loader, cfg, out):
    print("Student training started...")
    min_err = 10000 # Stores the best validation error so far.
    teacher.eval()  # Teacher model is forzen
    student.train() # Student model is set to training mode
    
    best_student = None
    
    # Using SGD optimizer for training the student model
    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    # Main training loop
    for epoch in range(cfg["student_training"]["epochs"]):
        student.train()
        running_loss = 0.0
        num_batches = 0
        
        # Training loop per batch
        for batch_data in train_loader:
            _, batch_img = batch_data
            batch_img = batch_img.cuda()

            # Feeding images for both teacher and student networks
            with torch.no_grad():
                # Teacher ouputs are treated as the target feature maps.
                t_feat = teacher(batch_img) # Teacher feature extraction (frozen)
            s_feat = student(batch_img)     # Student feature extraction (to be trained)

            loss =  0
            for i in range(len(t_feat)):
                # Both teacher and student features are L2-normalized (F.normalize).
                t_feat[i] = F.normalize(t_feat[i], dim=1)
                s_feat[i] = F.normalize(s_feat[i], dim=1)
                # The loss = average squared distance between the teacher and student features, summed over feature levels.
                loss += torch.sum((t_feat[i] - s_feat[i]) ** 2, 1).mean()

            print('[%d/%d] loss: %f' % (epoch, cfg["student_training"]["epochs"], loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = running_loss / num_batches
        print(f'Epoch [{epoch+1}/{cfg["student_training"]["epochs"]}] - Average Training Loss: {avg_train_loss:.6f}')

        # Runs the test() function on the validation set.
        err = get_error_map(teacher, student, val_loader)
        
        err_mean = err.mean()
        print('Valid Loss: {:.7f}'.format(err_mean.item()))
        if err_mean < min_err:
            min_err = err_mean
            save_name = os.path.join(out["student"], cfg["models"]["st_path"], 'student_best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': cfg["dataset"]["category"],
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)
            best_student = copy.deepcopy(student)
    print("Student training completed.")
    return best_student
    
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

def merge_boxes_touching_or_near(boxes, gap=4, iou_thresh=0.0):
    """
    Merge boxes that touch or are within 'gap' pixels (via gap-expansion overlap).
    Optionally also merge if IoU exceeds iou_thresh.
    boxes: list[(x,y,w,h), ...]
    """
    if not boxes:
        return []

    # Convert to x1,y1,x2,y2
    B = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes], dtype=np.int32)

    merged = True
    while merged:
        merged = False
        keep = []
        used = np.zeros(len(B), dtype=bool)
        for i in range(len(B)):
            if used[i]:
                continue
            xi1, yi1, xi2, yi2 = B[i]
            for j in range(i+1, len(B)):
                if used[j]:
                    continue
                xj1, yj1, xj2, yj2 = B[j]
                near = boxes_touch_or_near([xi1, yi1, xi2, yi2], [xj1, yj1, xj2, yj2], gap)
                ok_iou = (iou_thresh > 0.0) and (iou([xi1, yi1, xi2, yi2], [xj1, yj1, xj2, yj2]) >= iou_thresh)
                if near or ok_iou:
                    # union
                    xi1, yi1 = min(xi1, xj1), min(yi1, yj1)
                    xi2, yi2 = max(xi2, xj2), max(yi2, yj2)
                    used[j] = True
                    merged = True
            used[i] = True
            keep.append([xi1, yi1, xi2, yi2])
        B = np.array(keep, dtype=np.int32)

    # Back to x,y,w,h
    out = [(int(x1), int(y1), int(x2-x1), int(y2-y1)) for (x1, y1, x2, y2) in B.tolist()]
    return out

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

    file_path = os.path.join(out["calibration"],cfg.calibration_path)
    np.savez(file_path, mean=float(mean), std=float(std))
    print(f"[calib] saved μ={mean:.6g}, σ={std:.6g}")
    return mean, std