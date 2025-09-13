import copy
import os
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from evaluate import evaluate
from sklearn.model_selection import train_test_split
from glob import glob

from models.teacher import ResNet18_MS3
from data.mvtec_dataset import MVTecDataset
from data.data_utils import load_ground_truth

from config.config import load_args, load_config
from utils import *

def main():
    args = load_args()
    config = load_config(args.config)
    
    for key, value in config.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print_config_used(args)

    if args.mode == 'train':
        print("Training on category: ", args.category)
        #print("Dataset path: " ,glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.png')))
        image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'train', 'good', '*.png')))
        train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
        train_dataset = MVTecDataset(train_image_list, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        print("Training dataset loaded")
        val_dataset = MVTecDataset(val_image_list, transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        print("Validation dataset loaded")
        test_image_list = glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))
        test_dataset = MVTecDataset(test_image_list, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        print("Test dataset loaded")
    elif args.mode == 'test':
        print("Testing on category: ", args.category)
        test_neg_image_list = sorted(glob(os.path.join(args.mvtec_ad, args.category, 'test', 'good', '*.png')))
        test_pos_image_list = set(glob(os.path.join(args.mvtec_ad, args.category, 'test', '*', '*.png'))) - set(test_neg_image_list)
        test_pos_image_list = sorted(list(test_pos_image_list))
        test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
        test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
        test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
        test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)
    teacher.cuda()
    student.cuda()

    if args.mode == 'train':
        if args.checkpoint:
            print('loading model ' + args.checkpoint)
            saved_dict = torch.load(args.checkpoint)
            student.load_state_dict(saved_dict['state_dict'])
            best_student = copy.deepcopy(student)
        else:   
            best_student = train_val_student(teacher, student, train_loader, val_loader, args)
            
        test_err_map = get_error_map(teacher, best_student, test_loader)
        crop_images(test_err_map, test_loader, args)
        # triplet_learning(args)
    elif args.mode == 'test':
        saved_dict = torch.load(args.checkpoint)
        category = args.category
        gt = load_ground_truth(args.mvtec_ad, category)

        print('load ' + args.checkpoint)
        student.load_state_dict(saved_dict['state_dict'])

        pos = get_error_map(teacher, student, test_pos_loader)
        neg = get_error_map(teacher, student, test_neg_loader)

        scores = []
        for i in range(len(pos)):
            temp = cv2.resize(pos[i], (256, 256))
            scores.append(temp)
        for i in range(len(neg)):
            temp = cv2.resize(neg[i], (256, 256))
            scores.append(temp)

        scores = np.stack(scores)
        neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)
        gt_pixel = np.concatenate((gt, neg_gt), 0)
        gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)        

        pro = evaluate(gt_pixel, scores, metric='pro')
        auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
     

def crop_images(loss_map, loader, args):
    print("Starting cropping images...")
    img_dir = f"outputs/{args.category}/images"
    crops_dir = f"outputs/{args.category}/crops"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    
    print("[dbg] len(loader.dataset) =", len(loader.dataset))
    print("[dbg] loss_map.shape      =", getattr(loss_map, "shape", None))
    
    idx = 0
    for batch in loader:
        img_paths, _ = batch
        
        bs = len(img_paths)
        hm_batch = loss_map[idx: idx + bs]
        idx += bs
        
        for k, p in enumerate(img_paths):
            hm64 = hm_batch[k]
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: Unable to read image at {p}. Skipping.")
                continue
            
            H, W = img.shape[:2]
            hm_up = upscale_heatmap_to_image(hm64, (H, W))
            hm_gray = (hm_up * 255.0).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_gray, cv2.COLORMAP_JET)
            overlay  = cv2.addWeighted(img, 1.0, hm_color, 0.35, 0.0)
            
            mask, thr = threshold_heatmap(hm_up, method=args.threshold_method, percentile=float(args.threshold_value))
            # clean mask
            K_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # remove salt-noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_open)

            K_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))     # fuse close fragments
            mask = cv2.dilate(mask, K_dil, iterations=1)

            boxes = components_to_bboxes(mask, min_area=120, ignore_border=True)
            # --- merge touching/near boxes, then (optionally) expand ---
            boxes = merge_boxes_touching_or_near(boxes, gap=4, iou_thresh=0.0)
            boxes = expand_boxes(boxes, H, W, expand_ratio=0.12)  # 12% padding; set 0.0 to disable
            
            boxes_vis = draw_boxes(overlay, boxes, color=(0, 255, 0), thickness=2)
            os.makedirs(crops_dir, exist_ok=True)

            stem = Path(p).stem
            defect = Path(p).parent.name

            for bi, (x,y,w,h) in enumerate(boxes):
                x0, y0, x1, y1 = pad_box(x,y,w,h,H,W,pad_ratio=0.05)
                crop = img[y0:y1, x0:x1]
                cv2.imwrite(os.path.join(crops_dir, f"{defect}_{stem}_box{bi}.png"), crop)
            
            cv2.imwrite(os.path.join(img_dir, f"{defect}_{stem}_orig.png"), img)
            cv2.imwrite(os.path.join(img_dir, f"{defect}_{stem}_overlay.png"), overlay)
            cv2.imwrite(os.path.join(img_dir, f"{defect}_{stem}_mask.png"), mask)
            cv2.imwrite(os.path.join(img_dir, f"{defect}_{stem}_boxes.png"), boxes_vis)
                   
    print("Cropping images completed.")

def triplet_learning(args):
    print("Starting triplet learning...")
    triplet_model = ResNet18_MS3(pretrained=False)
    
    train_tf = transforms.Compose([
        transforms.Resize(256),  # keep aspect ratio
        transforms.RandomRotation(12, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_image_list = glob(os.path.join("outputs/Defect labeling.v1i.folder", 'train', '*', '*.jpg'))
    train_dataset = MVTecDataset(train_image_list, transform=train_tf)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    print("Triplet learning training dataset loaded")
    
    val_image_list = glob(os.path.join("outputs/Defect labeling.v1i.folder", 'valid', '*', '*.jpg'))
    val_dataset = MVTecDataset(train_image_list, transform=val_tf)
    val_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=False)
    print("Triplet learning validation dataset loaded")
    
    
    print("Triplet learning completed.")

def print_config_used(args):
    print("Configuration used:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("\n")

if __name__ == "__main__":
    main()