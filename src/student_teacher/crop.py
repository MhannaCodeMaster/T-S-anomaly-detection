import argparse
import datetime

import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.student_teacher.teacher import ResNet18_MS3
from src.student_teacher.st_eval import load_calibration_stats, load_datasets, get_error_map
from src.utils.utils import *

def main():
    args = load_args()

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)

    st_path = f'{args.st_path}/student_best.pth.tar'
    st_saved_dict = torch.load(st_path)
    student.load_state_dict(st_saved_dict['state_dict'])
    # Fetching calibration stats that were extracted during stduent training
    mean, std = load_calibration_stats(args.st_path)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225]),
    ])

    # loading image
    loader = load_datasets(args.img_path, transform)

    # Getting the error map
    err_map = get_error_map(teacher, student, loader)

    crop_images(err_map, loader, mean, std)


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
                # cv2.imwrite(os.path.join(f'{base_eval}/crops', f"{defect}_{stem}_box{bi}.png"), crop)
            
            
            # cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_orig.png"), img)
            # cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_overlay.png"), overlay)
            # cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_mask.png"), mask)
            # cv2.imwrite(os.path.join(f'{base_eval}/images', f"{defect}_{stem}_boxes.png"), boxes_vis)
            # print(f"images processed: [{idx + k + 1}/{total}]", end="\r")
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(img)
            axes[0].set_title('Original')
            axes[0].axis('off')

            axes[1].imshow(overlay)
            axes[1].set_title('Overlay')
            axes[1].axis('off')

            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('Mask')
            axes[2].axis('off')

            axes[3].imshow(boxes_vis)
            axes[3].set_title('Boxes')
            axes[3].axis('off')

            plt.tight_layout()
            plt.show()
            
        idx += bs

    print("\nCropping images completed.",end="\n")
    return crops, boxes, orig_img_path

def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--img_path", required=True, type=str, help="Path to the image")
    p.add_argument("--category", required=True, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--st_path", required=True, type=str, help="Path to the student model folder")
    
    #----- Optional args -----#
    p.add_argument("--h_th", required=True, default=95, type=float, help="Heatmap threshold")
    p.add_argument("--box_min_area", required=False, default=1000, type=int, help="Minimum box area")
    p.add_argument("--conf_score", required=False, default=0.1, type=float, help="Confidence score for NMS")
    p.add_argument("--nms_thr", required=False, default=0.4, type=float, help="NMS threshold")
    p.add_argument("--expand_box", required=False, default=0.1, type=float, help="Expand box %")
    p.add_argument("--tolerance", required=False, default=0.7, type=float, help="Box tolerance")
    p.add_argument("--merge_box", required=False, default=0, type=str, help="Enable merge boxes (0 or 1)")
    p.add_argument("--merge_gap", required=False, default=1, type=float, help="Merge gap between boxes")

    args = p.parse_args()
    return args


if __name__ == "__main__":
    main()