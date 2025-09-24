import os
import copy
from types import SimpleNamespace

import torch
from torchvision import transforms
import numpy as np

from src.student_teacher.teacher import ResNet18_MS3
from src.data.data_utils import *
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
        transforms.Resize([256, 256]),
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
    crop_images(test_err_map, test_loader, mean, std, args)
    # triplet_eval()

def crop_images(loss_map, loader, mean, std, cfg):
    print("Starting cropping images...",end='\n')  
    total = len(loader.dataset)
    print(f"images processed: [0/{total}]",end="\r")
    idx = 0
    for batch in loader:
        img_paths, _ = batch
        
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
            
            mask, thr = threshold_heatmap(hm_z, method='percentile', percentile=94.5)
            # clean mask
            K_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # remove salt-noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_open)

            K_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))     # fuse close fragments
            mask = cv2.dilate(mask, K_dil, iterations=1)

            boxes = components_to_bboxes(mask, min_area=700, ignore_border=True)
            # Run NMS
            scores = get_box_scores(boxes, hm_z, mode='mean')
            score_thr = score_thr # confidence threshold
            nms_thr   = thr
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]

            boxes = expand_boxes(boxes, H, W, expand_ratio=0.0)  # set 0.0 to disable
            boxes = remove_nested_boxes(boxes, tolerance=0.7)
            
            boxes_vis = draw_boxes(overlay, boxes, color=(0, 0, 255), thickness=2)

            stem = Path(p).stem
            defect = Path(p).parent.name

            for bi, (x,y,w,h) in enumerate(boxes):
                x0, y0, x1, y1 = pad_box(x,y,w,h,H,W,pad_ratio=0.05)
                crop = img[y0:y1, x0:x1]
                cv2.imwrite(os.path.join('/kaggle/working/eval/crops', f"{defect}_{stem}_box{bi}.png"), crop)
            
            cv2.imwrite(os.path.join('/kaggle/working/eval/images', f"{defect}_{stem}_orig.png"), img)
            cv2.imwrite(os.path.join('/kaggle/working/eval/images', f"{defect}_{stem}_overlay.png"), overlay)
            cv2.imwrite(os.path.join('/kaggle/working/eval/images', f"{defect}_{stem}_mask.png"), mask)
            cv2.imwrite(os.path.join('/kaggle/working/eval/images', f"{defect}_{stem}_boxes.png"), boxes_vis)
            print(f"images processed: [{idx + k + 1}/{total}]", end="\r")
        idx += bs
                   
    print("\nCropping images completed.",end="\n")


def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Dataset root path")
    p.add_argument("--category", required=True, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--st_path", required=True, type=str, help="Student model path")
    p.add_argument("--tl_path", required=True, type=str, help="Triplet model path")
    p.add_argument("--calibration", required=True, type=str, help="Calibration path")
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

if __name__ == '__main__':
    main()