import torch
import torch.optim
import torch.nn.functional as F
import copy
import os
import cv2
import numpy as np

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

def train_val_student(teacher, student, train_loader, val_loader, args):
    print("Student training started...")
    min_err = 10000 # Stores the best validation error so far.
    teacher.eval()  # Teacher model is forzen
    student.train() # Student model is set to training mode
    
    best_student = None
    
    # Using SGD optimizer for training the student model
    optimizer = torch.optim.SGD(student.parameters(), 0.4, momentum=0.9, weight_decay=1e-4)
    # Main training loop
    for epoch in range(args.epochs):
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

            print('[%d/%d] loss: %f' % (epoch, args.epochs, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = running_loss / num_batches
        print(f'Epoch [{epoch+1}/{args.epochs}] - Average Training Loss: {avg_train_loss:.6f}')

        # Runs the test() function on the validation set.
        err = get_error_map(teacher, student, val_loader)
        
        err_mean = err.mean()
        print('Valid Loss: {:.7f}'.format(err_mean.item()))
        if err_mean < min_err:
            min_err = err_mean
            save_name = os.path.join(args.model_save_path, args.category, 'best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': args.category,
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

def components_to_bboxes(mask: np.ndarray, min_area: int = 20, ignore_border: bool = True):
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
        if ignore_border:
            touches_border = (x == 0) or (y == 0) or (x + w >= W) or (y + h >= H)
            if touches_border:
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