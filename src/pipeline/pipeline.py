import os
import copy
from types import SimpleNamespace

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from sklearn.metrics import f1_score, roc_auc_score


from src.student_teacher.teacher import ResNet18_MS3
from src.data.data_utils import *
from src.data.datasets import *
from src.triplet.triplet import *

from conf.config import *
from src.utils.utils import *
from src.paths import get_paths



def main():
    args = load_args()
    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
    triplet = TripletEmbedder(pretrained=False)

    teacher.cuda()
    student.cuda()

    # 1) Student (teacher–student) uses 256×256
    st_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
    ])

    tl_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
    ])

    st_saved_dict = torch.load(args.st_path)
    student.load_state_dict(st_saved_dict['state_dict'])
    student.cuda()
    
    tl_saved_dict = torch.load(args.tl_path)
    triplet.load_state_dict(tl_saved_dict['state_dict'])
    triplet.cuda()
    
    test_loader = load_test_datasets(st_transform, args)
    mean, std = load_calibration_stats(args.calibration)
    
    all_true, all_pred, all_score = [], [], []
    for batch in test_loader:
        paths, x = batch
        y = [0 if Path(p).parent.name == "good" else 1 for p in paths]  # labels
        y = torch.tensor(y, device=x.device)
        x = x.cuda(non_blocking=True)
        
        test_err_map = get_error_map_v1(teacher, student, [(paths,x)])                
        # crops, boxes, image_paths = crop_images(test_err_map, [(x,y,_,paths)], mean, std, args)
        # print('Image used:', image_paths)
        
        for (crops, boxes, img_path) in crop_images_iter(test_err_map, [(paths, x)], mean, std, args):
            res2 = triplet_classifier_knn(triplet, tl_transform, boxes, [img_path], crops, args,
                                        k=30, tau=0.35, SIM_MIN=0.15, device='cuda')
            
            label = get_mvtec_label(img_path)
            all_true.append(label)
            all_pred.append(res2["image_pred"])
            all_score.append(res2["image_score"])
        
        # res1 = triplet_classifer(triplet, tl_transform, boxes, image_paths, crops, args)
        # res2 = triplet_classifier_knn( triplet, tl_transform, boxes, image_paths, crops, args, k=30, tau=0.35, SIM_MIN=0.15, device='cuda')
        # res_proto = triplet_classifier_proto(triplet, tl_transform, boxes, image_paths, crops, args,beta=10.0, out_path="result_proto.png")
    
    f1 = f1_score(all_true, all_pred)
    print(f"F1 = {f1:.4f}")

    if len(set(all_true)) == 2:                 # both classes present
        auroc = roc_auc_score(all_true, all_score)
        print(f"AUROC = {auroc:.4f}")
    else:
        print("AUROC skipped: only one class present in y_true.")

def get_mvtec_label(img_path: str) -> int:
    """
    For MVTec-AD test set:
    - returns 0 if parent folder is 'good'
    - returns 1 otherwise (any defect type)
    """
    print(img_path)
    label = 0 if "good" in str(img_path).lower() else 1
    return label

def crop_images(loss_map, loader, mean, std, args):
    print("Starting cropping images...",end='\n')  
    total = len(loader.dataset)

    #--------- CONFIG -----------#
    CATEGORY = args.category
    ENABLE_BOX_MERGE = args.merge_box
    GAP_PX = args.merge_gap  # or use args.merge_gap if you add it
    HM_THR = args.h_th
    BOX_MIN_AREA = args.box_min_area
    SCRORE_THR = args.conf_score # confidence threshold
    NMS_THR   = args.nms_thr
    EXPAND_BOX = args.expand_box
    TOLERANCE = args.tolerance
    #----------------------------#

    print(f"images processed: [0/{total}]",end="\r")
    idx = 0
    orig_img_path = []
    for batch in loader:
        img_paths, _ = batch
        orig_img_path = img_paths
        bs = len(img_paths)
        hm_batch = loss_map[idx: idx + bs]
        for k, p in enumerate(img_paths):
            hm64 = hm_batch[k]
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: Unable to read image at {p}. Skipping.",end="\n")
                continue
            
            H, W = img.shape[:2]
            hm_up = upscale_heatmap_to_image(hm64, (H, W))
            hm_z = zscore_calibrate(hm_up, mean, std)
            
            hm_gray = (hm_z * 255.0).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_gray, cv2.COLORMAP_JET)
            overlay  = cv2.addWeighted(img, 1.0, hm_color, 0.35, 0.0)
            
            mask, thr = threshold_heatmap(hm_z, method='percentile', percentile=HM_THR)
            #----------- clean mask -----------#
            K_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # remove salt-noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_open)

            K_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))     # fuse close fragments
            mask = cv2.dilate(mask, K_dil, iterations=1)

            K_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_close)  # joins nearby blobs

            #----------- Building boxes from mask ------------#
            boxes = components_to_bboxes(mask, min_area=BOX_MIN_AREA, ignore_border=True)

            #----------- Applying NMS algo -----------#
            scores = get_box_scores(boxes, hm_z, mode='mean')
            indices = cv2.dnn.NMSBoxes(boxes, scores, SCRORE_THR, NMS_THR)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]

            #----------- Modifying boxes --------------#
            boxes = expand_boxes(boxes, H, W, expand_ratio=EXPAND_BOX)  # set 0.0 to disable
            boxes = remove_nested_boxes(boxes, tolerance=TOLERANCE)
            if ENABLE_BOX_MERGE == "1":
                boxes = merge_touching_boxes_xywh(boxes, gap=GAP_PX, iou_thr=0.0, max_iters=5)
            
            boxes_vis = draw_boxes(overlay, boxes, color=(0, 0, 255), thickness=2)

            stem = Path(p).stem
            defect = Path(p).parent.name

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            runtime = f"{ts}_{CATEGORY.lower()}"

            base_eval = f"/kaggle/working/eval/{CATEGORY.lower()}/{runtime}"
            os.makedirs(os.path.join(base_eval, "crops"), exist_ok=True)
            os.makedirs(os.path.join(base_eval, "images"), exist_ok=True)

            crops = []
            for bi, (x,y,w,h) in enumerate(boxes):
                x0, y0, x1, y1 = pad_box(x,y,w,h,H,W,pad_ratio=0.05)
                crop = img[y0:y1, x0:x1]
                crops.append(crop)
                cv2.imwrite(os.path.join(f'{base_eval}/crops', f"{defect}_{stem}_box{bi}.png"), crop)
            
            cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_orig.png"), img)
            cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_overlay.png"), overlay)
            cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_mask.png"), mask)
            cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_boxes.png"), boxes_vis)
            print(f"images processed: [{idx + k + 1}/{total}]", end="\r")
        idx += bs

    print("\nCropping images completed.",end="\n")
    return crops, boxes, orig_img_path

def crop_images_iter(loss_map, loader_or_batch, mean, std, args, save_debug=True):
    """
    Yields per-image: (crops:list[np.ndarray], boxes:list[xywh], image_path:str)

    loader_or_batch must yield (paths, images).
    """
    # detect iterable & total
    if hasattr(loader_or_batch, "dataset"):
        iterable = loader_or_batch
        total = len(loader_or_batch.dataset)
    elif isinstance(loader_or_batch, (list, tuple)) and len(loader_or_batch) == 2:
        iterable = [loader_or_batch]
        total = loader_or_batch[1].size(0)
    elif isinstance(loader_or_batch, list) and loader_or_batch and len(loader_or_batch[0]) == 2:
        iterable = loader_or_batch
        total = sum(batch[1].size(0) for batch in loader_or_batch)
    else:
        raise ValueError("crop_images_iter expects a DataLoader or (paths, images) batch.")

    #--------- CONFIG -----------#
    CATEGORY       = args.category
    ENABLE_BOX_MERGE = args.merge_box
    GAP_PX        = args.merge_gap
    HM_THR        = args.h_th
    BOX_MIN_AREA  = args.box_min_area
    SCORE_THR     = args.conf_score
    NMS_THR       = args.nms_thr
    EXPAND_BOX    = args.expand_box
    TOLERANCE     = args.tolerance
    #----------------------------#

    # One run directory for all images (cleaner)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"/kaggle/working/eval/{CATEGORY.lower()}/{ts}_{CATEGORY.lower()}"
    if save_debug:
        os.makedirs(os.path.join(run_dir, "crops"),  exist_ok=True)
        os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)

    print(f"Starting cropping images… [0/{total}]", end="\r")
    idx_global = 0

    for paths, _ in iterable:                    # <<== (paths, images)
        bs = len(paths)
        hm_batch = loss_map[idx_global: idx_global + bs]

        for k, p in enumerate(paths):
            hm64 = hm_batch[k]
            img = cv2.imread(p)
            if img is None:
                print(f"\n[WARN] Unable to read image: {p} — skipping.")
                idx_global += 1
                continue

            H, W = img.shape[:2]
            hm_up = upscale_heatmap_to_image(hm64, (H, W))
            hm_z  = zscore_calibrate(hm_up, mean, std)

            # threshold -> mask
            mask, thr = threshold_heatmap(hm_z, method='percentile', percentile=HM_THR)

            # clean mask
            K_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_open)
            K_dil   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask    = cv2.dilate(mask, K_dil, iterations=1)
            K_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_close)

            # boxes
            boxes   = components_to_bboxes(mask, min_area=BOX_MIN_AREA, ignore_border=True)
            scores  = get_box_scores(boxes, hm_z, mode='mean')
            indices = cv2.dnn.NMSBoxes(boxes, scores, SCORE_THR, NMS_THR)
            if len(indices) > 0:
                boxes = [boxes[i] for i in indices.flatten()]

            boxes = expand_boxes(boxes, H, W, expand_ratio=EXPAND_BOX)
            boxes = remove_nested_boxes(boxes, tolerance=TOLERANCE)
            if str(ENABLE_BOX_MERGE) == "1":
                boxes = merge_touching_boxes_xywh(boxes, gap=GAP_PX, iou_thr=0.0, max_iters=5)

            # crops
            crops = []
            for bi, (x, y, w, h) in enumerate(boxes):
                x0, y0, x1, y1 = pad_box(x, y, w, h, H, W, pad_ratio=0.05)
                crops.append(img[y0:y1, x0:x1])

            # debug saves
            if save_debug:
                stem   = Path(p).stem
                defect = Path(p).parent.name

                # visuals
                hm_gray  = (np.clip(hm_z, 0, None) / (hm_z.max() + 1e-6) * 255.0).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_gray, cv2.COLORMAP_JET)
                overlay  = cv2.addWeighted(img, 1.0, hm_color, 0.35, 0.0)
                boxes_vis = draw_boxes(overlay, boxes, color=(0, 0, 255), thickness=2)

                # save
                for bi, crop in enumerate(crops):
                    cv2.imwrite(os.path.join(run_dir, "crops", f"{defect}_{stem}_box{bi}.png"), crop)
                cv2.imwrite(os.path.join(run_dir, "images", f"{defect}_{stem}_orig.png"),    img)
                cv2.imwrite(os.path.join(run_dir, "images", f"{defect}_{stem}_overlay.png"), overlay)
                cv2.imwrite(os.path.join(run_dir, "images", f"{defect}_{stem}_mask.png"),    mask)
                cv2.imwrite(os.path.join(run_dir, "images", f"{defect}_{stem}_boxes.png"),   boxes_vis)

            idx_global += 1
            print(f"Cropping images… [{idx_global}/{total}]", end="\r")

            # yield per-image
            yield crops, boxes, p

    print("\nCropping images completed.")

def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Dataset root path")
    p.add_argument("--category", required=True, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--st_path", required=True, type=str, help="Student model path")
    p.add_argument("--h_th", required=False, default=99, type=float, help="Heatmap_threshold")
    p.add_argument("--tl_path", required=True, type=str, help="Triplet model path")
    p.add_argument("--calibration", required=True, type=str, help="Calibration path")
    p.add_argument("--emd_gal", required=True, type=str, help="Saved embeddings gallery")
    
    p.add_argument("--pred_thr", required=False, default=0.8, type=float, help="triplet model prediction threshold")
    p.add_argument("--box_min_area", required=False, default=600, type=int, help="Minimum box area")
    p.add_argument("--conf_score", required=False, default=0.1, type=float, help="Confidence score")
    p.add_argument("--nms_thr", required=False, default=0.4, type=float, help="NMS threshold")
    p.add_argument("--expand_box", required=False, default=0.1, type=float, help="Expand box %")
    p.add_argument("--tolerance", required=False, default=0.7, type=float, help="Box tolerance")
    p.add_argument("--merge_gap", required=False, default=1, type=float, help="Merge gap between boxes")
    p.add_argument("--merge_box", required=False, default=0, type=str, help="Enable merge boxes (0 or 1)")

    args = p.parse_args()
    return args

def load_test_datasets(transform, args):
    print("Loading Test dataset...")
    image_list = sorted(glob(os.path.join(args.dataset, args.category, 'test', '*', '*.png')))
    #test_image_list = [random.choice(image_list)]
    test_dataset = MVTecDataset(image_list, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    print("Loading test dataset completed.")
    return test_loader

def preprocess_crops(crops, transform):
    tensors = []
    for crop in crops:
        # convert BGR (OpenCV) → RGB (PIL)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensors.append(transform(pil_img))
    if len(tensors) == 0:
        return None
    return torch.stack(tensors)   # shape (N, C, H, W)

@torch.no_grad()
def embed_crops(model, crops, transform, device="cuda"):
    x = preprocess_crops(crops, transform)
    if x is None: 
        return None, None
    x = x.to(device)
    z = model(x)                  # (N, D)
    z = F.normalize(z, p=2, dim=1)
    return z, x

@torch.no_grad()
def triplet_classifer(model, transform, boxes, image_paths, crops, args,
                      k=5, device='cuda', tau=0.07, prior_correction=True):
    """
    Mixed gallery (OK + NOT_OK) k-NN in cosine space.
    - tau: temperature for softmax on similarities (lower => sharper).
    - prior_correction: reweights neighbors by inverse class frequency.
    """
    assert len(image_paths) > 0
    PRED_THRESHOLD = float(args.pred_thr)
    orig_image_path = image_paths[0]

    # ---- Load gallery ----
    gal_pkg = torch.load(args.emd_gal, map_location='cpu', weights_only=False)
    Zg = torch.as_tensor(gal_pkg["embeddings"], dtype=torch.float32)   # [N,D]
    yg = torch.as_tensor(gal_pkg["labels"], dtype=torch.long)          # [N] (0=OK, 1=DEFECT)

    # normalize (cosine geometry)
    Zg = F.normalize(Zg, dim=1)
    N  = Zg.shape[0]
    assert N > 0, "Gallery is empty."

    # class priors (for imbalance correction)
    if prior_correction:
        n_ok  = max(1, int((yg==0).sum().item()))
        n_def = max(1, int((yg==1).sum().item()))
        w_ok, w_def = 1.0/n_ok, 1.0/n_def
    else:
        w_ok = w_def = 1.0

    # ---- Embed crops ----
    model.eval()
    z, _ = embed_crops(model, crops, transform, device=device)  # z: [B,D]
    z = F.normalize(z, dim=1)

    # ---- Move to device ----
    Zg = Zg.to(device, non_blocking=True)
    yg = yg.to(device, non_blocking=True)

    # ---- k-NN with cosine similarity ----
    # sim[b,n] = z_b · Zg_n  in [-1,1]; larger => more similar
    sim = z @ Zg.t()                                 # [B,N]
    kk  = min(int(k), N)
    sim_k, idx = torch.topk(sim, k=kk, dim=1, largest=True, sorted=False)   # [B,kk]
    y_k  = yg[idx]                                   # [B,kk]

    # soft weights in neighbor set
    w = torch.softmax(sim_k / tau, dim=1)            # [B,kk]

    # class-weighted fraction of defect neighbors
    class_w = torch.where(y_k==1, torch.tensor(w_def, device=w.device), torch.tensor(w_ok, device=w.device))
    w_adj   = w * class_w
    w_adj   = w_adj / (w_adj.sum(dim=1, keepdim=True) + 1e-8)
    defect_frac = (w_adj * (y_k==1).float()).sum(dim=1)   # [B] in [0,1]

    # ---- Decisions ----
    crop_scores = defect_frac.tolist()
    crop_preds  = (defect_frac >= PRED_THRESHOLD).long().tolist()
    image_score = float(defect_frac.max().item())
    image_pred  = int(image_score >= PRED_THRESHOLD)

    # ---- Visualization ----
    img = cv2.imread(orig_image_path)
    if img is not None:
        for (x, y, w_box, h_box), pred, score in zip(boxes, crop_preds, crop_scores):
            color = (0,0,255) if pred==1 else (0,255,0)
            cv2.rectangle(img, (x,y), (x+w_box,y+h_box), color, 2)
            cv2.putText(img, f"{score:.2f}", (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite("result1.png", img)

    return {
        "crop_scores": crop_scores,
        "crop_preds":  crop_preds,
        "image_score": image_score,
        "image_pred":  image_pred,
        "idx_topk":    idx.detach().cpu().numpy(),
        "sim_topk":    sim_k.detach().cpu().numpy()
    }

@torch.no_grad()
def triplet_classifier_knn(
    model, transform, boxes, image_paths, crops, args,
    k=15, device="cuda", tau=0.25, prior_correction=True,
    SIM_MIN=0.15,   # ignore neighbors with cosine below this
):
    PRED_THRESHOLD = float(args.pred_thr)
    orig_image_path = image_paths[0]

    gal_pkg = torch.load(args.emd_gal, map_location="cpu", weights_only=False)
    Zg = torch.as_tensor(gal_pkg["embeddings"], dtype=torch.float32)   # [N,D]
    yg = torch.as_tensor(gal_pkg["labels"], dtype=torch.long)          # [N] (0 OK, 1 DEF)
    Zg = F.normalize(Zg, dim=1)

    model.eval()
    z, _ = embed_crops(model, crops, transform, device=device)  # z: [B,D]
    z = F.normalize(z, dim=1)

    Zg = Zg.to(device, non_blocking=True)
    yg = yg.to(device, non_blocking=True)

    # priors (optional)
    if prior_correction:
        n_ok  = max(1, int((yg == 0).sum().item()))
        n_def = max(1, int((yg == 1).sum().item()))
        w_ok, w_def = 1.0 / n_ok, 1.0 / n_def
    else:
        w_ok = w_def = 1.0

    sim = z @ Zg.t()                                    # [B,N] cosine
    kk  = min(int(k), Zg.size(0))
    sim_k, idx = torch.topk(sim, k=kk, dim=1, largest=True, sorted=False)  # [B,kk]
    y_k  = yg[idx]                                      # [B,kk]

    # mask out weak neighbors (too dissimilar)
    valid = (sim_k >= SIM_MIN).float()                  # [B,kk]
    # softmax over only valid neighbors
    logits = sim_k / max(1e-6, tau)
    logits = logits - (1.0 - valid) * 1e9               # -inf for invalid
    w = torch.softmax(logits, dim=1) * valid            # re-zero invalid
    # renormalize
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    class_w = torch.where(y_k == 1,
                          torch.tensor(w_def, device=w.device),
                          torch.tensor(w_ok,  device=w.device))
    w_adj = w * class_w
    w_adj = w_adj / (w_adj.sum(dim=1, keepdim=True) + 1e-8)

    # per-class masses
    m_def = (w_adj * (y_k == 1).float()).sum(dim=1)     # [B]
    m_ok  = (w_adj * (y_k == 0).float()).sum(dim=1)
    # if nothing valid, masses become 0; we’ll detect that and fall back
    total_mass = m_def + m_ok
    defect_frac = torch.where(total_mass > 0, m_def / (total_mass + 1e-8), torch.zeros_like(m_def))

    crop_scores = defect_frac.tolist()
    crop_preds  = (defect_frac >= PRED_THRESHOLD).long().tolist()
    image_score = float(defect_frac.max().item())
    image_pred  = int(image_score >= PRED_THRESHOLD)

    # visualize
    img = cv2.imread(orig_image_path)
    if img is not None:
        for (x, y, w_box, h_box), pred, score in zip(boxes, crop_preds, crop_scores):
            color = (0,0,255) if pred==1 else (0,255,0)
            cv2.rectangle(img, (x,y), (x+w_box,y+h_box), color, 2)
            cv2.putText(img, f"{score:.2f}", (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite("result_knn.png", img)

    return {
        "crop_scores": crop_scores,
        "crop_preds":  crop_preds,
        "image_score": image_score,
        "image_pred":  image_pred,
        "idx_topk":    idx.detach().cpu().numpy(),
        "sim_topk":    sim_k.detach().cpu().numpy(),
        "valid_frac":  float((valid.sum().item()) / max(1, valid.numel()))
    }

@torch.no_grad()
def build_prototypes(embeddings: torch.Tensor, labels: torch.Tensor):
    # embeddings: [N,D] (already normalized), labels: [N]
    proto_ok  = F.normalize(embeddings[labels==0].mean(dim=0, keepdim=True), dim=1)   # [1,D]
    proto_def = F.normalize(embeddings[labels==1].mean(dim=0, keepdim=True), dim=1)   # [1,D]
    return proto_ok, proto_def

@torch.no_grad()
def triplet_classifier_proto(model, transform, boxes, image_paths, crops, args,
                                 device="cuda", beta=10.0, out_path="result_proto.png"):
    """
    Prototype (centroid) classifier in cosine space + draws boxes on the image.
    - Builds two class prototypes (OK / DEFECT) from the gallery.
    - Scores each crop by sigmoid(beta * (cos_def - cos_ok)).
    - Saves an annotated image to out_path.
    """
    thr = float(args.pred_thr)
    orig_image_path = image_paths[0]

    # ---- Load gallery and build prototypes ----
    gal_pkg = torch.load(args.emd_gal, map_location="cpu", weights_only=False)
    Zg = torch.as_tensor(gal_pkg["embeddings"], dtype=torch.float32)   # [N,D]
    yg = torch.as_tensor(gal_pkg["labels"], dtype=torch.long)          # [N]
    Zg = F.normalize(Zg, dim=1)

    proto_ok  = F.normalize(Zg[yg==0].mean(dim=0, keepdim=True), dim=1)   # [1,D]
    proto_def = F.normalize(Zg[yg==1].mean(dim=0, keepdim=True), dim=1)   # [1,D]
    proto_ok  = proto_ok.to(device)
    proto_def = proto_def.to(device)

    # ---- Embed crops ----
    model.eval()
    z, _ = embed_crops(model, crops, transform, device=device)  # [B,D]
    z = F.normalize(z, dim=1)

    # ---- Score ----
    cos_ok  = (z @ proto_ok.t()).squeeze(1)         # [B]
    cos_def = (z @ proto_def.t()).squeeze(1)
    logits  = beta * (cos_def - cos_ok)
    prob_def = torch.sigmoid(logits)                # [B] defect probability

    crop_scores = prob_def.tolist()
    crop_preds  = (prob_def >= thr).long().tolist()
    image_score = float(prob_def.max().item())
    image_pred  = int(image_score >= thr)

    # ---- Visualization ----
    img = cv2.imread(orig_image_path)
    if img is not None:
        for (x, y, w_box, h_box), pred, score in zip(boxes, crop_preds, crop_scores):
            color = (0,0,255) if pred==1 else (0,255,0)
            cv2.rectangle(img, (x,y), (x+w_box,y+h_box), color, 2)
            cv2.putText(img, f"{score:.2f}", (x, max(0,y-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(out_path, img)

    return {
        "crop_scores": crop_scores,
        "crop_preds":  crop_preds,
        "image_score": image_score,
        "image_pred":  image_pred,
        "cos_ok":      cos_ok.detach().cpu().numpy(),
        "cos_def":     cos_def.detach().cpu().numpy(),
        "out_path":    out_path,
    }


if __name__ == '__main__':
    main()