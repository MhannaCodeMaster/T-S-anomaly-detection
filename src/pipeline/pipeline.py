import os
import copy
from types import SimpleNamespace

import torch
from torchvision import transforms
import numpy as np
from PIL import Image


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

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    st_saved_dict = torch.load(args.st_path)
    student.load_state_dict(st_saved_dict['state_dict'])
    student.cuda()
    
    tl_saved_dict = torch.load(args.tl_path)
    triplet.load_state_dict(tl_saved_dict['state_dict'])
    triplet.cuda()
    
    test_loader = load_test_datasets(transform, args)
    mean, std = load_calibration_stats(args.calibration)
    test_err_map = get_error_map(teacher, student, test_loader)                
    crops, boxes, image_paths = crop_images(test_err_map, test_loader, mean, std, args)
    print('Image used:', image_paths)
    res = triplet_classifer(triplet, transform, boxes, image_paths, crops, args)

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
    test_image_list = [random.choice(image_list)]
    test_dataset = MVTecDataset(test_image_list, transform=transform)
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

def triplet_classifer(model, transform, boxes, image_paths, crops, args, k=5, device='cuda'):
    orig_image_path = image_paths[0]
    PRED_THRESHOLD = args.pred_thr
    
    gal_pkg = torch.load(args.emd_gal, map_location='cpu', weights_only=False)
    Zg = gal_pkg["embeddings"]              # [N, D], L2-normalized
    yg = gal_pkg["labels"]           # [N], 0=OK, 1=DEFECT
    
    Zg = torch.as_tensor(Zg, dtype=torch.float32)  # [N, D]
    yg = torch.as_tensor(yg, dtype=torch.long)     # [N]  (0=OK, 1=DEFECT)
    
    Zg = torch.nn.functional.normalize(Zg, p=2, dim=1)
    
    z, x_tensor = embed_crops(model, crops, transform, device=device)  # z: [B,D]
    Zg = Zg.to(device)
    # k-NN against gallery (distance-weighted)
    # 3) k-NN scoring (distance-weighted fraction of defect neighbors)
    d   = torch.cdist(z, Zg)                                    # [B,N]
    kk  = min(k, Zg.shape[0])
    d_k, idx = torch.topk(d, k=kk, dim=1, largest=False)        # [B,kk]
    
    yg = yg.to(device)
    y_k = yg[idx]                                               # [B,kk]

    w = 1.0 / (d_k + 1e-6)
    w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
    defect_frac = (w * (y_k == 1).float()).sum(dim=1)           # [B] in [0,1]

    # 4) Decide defect crops + image score
    crop_preds = (defect_frac >= PRED_THRESHOLD).long().tolist()
    crop_scores = defect_frac.tolist()
    image_score = float(defect_frac.max().item())
    image_pred  = int(image_score >= PRED_THRESHOLD)

    # 5) Draw visualization
    img = cv2.imread(orig_image_path)
    if img is None:
        print(f"[warn] Could not read {orig_image_path} to draw boxes.")

    for (x, y, w, h), pred, score in zip(boxes, crop_preds, crop_scores):
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)  # red defect, green ok (BGR)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{score:.2f}", (x, max(0, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite("result.png", img)


if __name__ == '__main__':
    main()