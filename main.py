import os
from pathlib import Path
import copy

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from glob import glob

from models.teacher import ResNet18_MS3
from models.triplet import TripletEmbedder
from data.datasets import MVTecDataset
from data.data_utils import *
from evaluate import evaluate

from conf.config import *
from utils import *

def main():
    try:
        cfg = get_config()
        out = resolve_paths(cfg)
        save_config(cfg, out['base'])
        
        np.random.seed(0)
        torch.manual_seed(0)
        
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if cfg["mode"] == "train":
            print("Training on category: ", cfg["dataset"]["category"])
        elif cfg["mode"] == "test":
            print("Testing on category: ", cfg["dataset"]["category"])
            test_neg_loader, test_pos_loader = load_test_datasets(transform, cfg)

        if cfg["mode"] == "train":
            student = ResNet18_MS3(pretrained=False)
            mean, std = 0.0, 0.0
            
            if cfg["student"]["train"] == "true":
                teacher = ResNet18_MS3(pretrained=True)
                teacher.cuda()
                student.cuda()
                # train student model
                train_loader, val_loader = load_st_train_datasets(transform, cfg)
                student = train_student(teacher, student, train_loader, val_loader, cfg, out)
                mean, std = compute_train_calibration_stats(teacher, student, train_loader, cfg, out, device="cuda")
            elif cfg["student"]["checkpoint"]: # Load student model from checkpoint
                try:
                    print('loading model ' + cfg['student']['checkpoint'])
                    saved_dict = torch.load(cfg["student"]["checkpoint"])
                    student.load_state_dict(saved_dict['state_dict'])
                    student.cuda()
                    mean, std = load_calibration_stats(cfg["student"]["calibration"])
                except Exception as e:
                    print(f"Error loading student model from {cfg['student']['calibration']}: {e}")
                    raise
            else:
                print("Error: No student training or checkpoint provided.")
                return
            
            if cfg["triplet"]["train"] == "true":
                triplet = TripletEmbedder(pretrained=True)
                triplet.cuda()
                train_loader, val_loader = load_tl_datasets(cfg, out)
                train_triplet(triplet, train_loader, val_loader, cfg, out)
            
            #test_err_map = get_error_map(teacher, best_student, test_loader)                
            # crop_images(test_err_map, test_loader, mean, std, cfg, out)
            # triplet_learning(args)
        elif cfg["mode"] == "test":
            ST_CHECKPOINT = cfg["student"]["checkpoint"]
            DATASETPATH = cfg["dataset"]["mvtec_path"]
            CATEGORY = cfg["dataset"]["category"]
            saved_dict = torch.load(ST_CHECKPOINT)
            gt = load_ground_truth(DATASETPATH, CATEGORY)

            print('load ' + ST_CHECKPOINT)
            student = ResNet18_MS3(pretrained=False)
            student.load_state_dict(saved_dict['state_dict'])

        #     pos = get_error_map(teacher, student, test_pos_loader)
        #     neg = get_error_map(teacher, student, test_neg_loader)

        #     scores = []
        #     for i in range(len(pos)):
        #         temp = cv2.resize(pos[i], (256, 256))
        #         scores.append(temp)
        #     for i in range(len(neg)):
        #         temp = cv2.resize(neg[i], (256, 256))
        #         scores.append(temp)

        #     scores = np.stack(scores)
        #     neg_gt = np.zeros((len(neg), 256, 256), dtype=np.bool)
        #     gt_pixel = np.concatenate((gt, neg_gt), 0)
        #     gt_image = np.concatenate((np.ones(pos.shape[0], dtype=np.bool), np.zeros(neg.shape[0], dtype=np.bool)), 0)        

        #     pro = evaluate(gt_pixel, scores, metric='pro')
        #     auc_pixel = evaluate(gt_pixel.flatten(), scores.flatten(), metric='roc')
        #     auc_image_max = evaluate(gt_image, scores.max(-1).max(-1), metric='roc')
        #     print('Catergory: {:s}\tPixel-AUC: {:.6f}\tImage-AUC: {:.6f}\tPRO: {:.6f}'.format(category, auc_pixel, auc_image_max, pro))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def train_student(teacher, student, train_loader, val_loader, cfg, out):
    print("Student training started...")
    LR = float(cfg['student']['lr'])
    MOMENTUM = float(cfg['student']['momentum'])
    WGT_DECAY = float(cfg['student']['weight_decay'])
    EPOCHS = int(cfg['student']['epochs'])
    min_err = 10000 # Stores the best validation error so far.
    teacher.eval()  # Teacher model is forzen
    student.train() # Student model is set to training mode
    
    best_student = None
    
    # Using SGD optimizer for training the student model
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WGT_DECAY)
    # Main training loop

    for epoch in range(EPOCHS):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = running_loss / num_batches

        # get the error map using the validation set.
        err = get_error_map(teacher, student, val_loader)
        
        err_mean = err.mean()
        if err_mean < min_err:
            min_err = err_mean
            save_name = os.path.join(out["student"],'student_best.pth.tar')
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': cfg["dataset"]["category"],
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)
            best_student = copy.deepcopy(student)
        print(f"Epoch: [{epoch+1}/{cfg['student']['epochs']}] - Avg training loss: {avg_train_loss:.6f} - Validation loss: {err_mean.item():.7f}",end="\r")
        
    print("\nStudent training completed.")
    return best_student

def crop_images(loss_map, loader, mean, std, cfg, out):
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
            
            mask, thr = threshold_heatmap(hm_z, method=cfg["heatmap_threshold"]["method"], percentile=float(cfg["heatmap_threshold"]["value"]))
            # clean mask
            K_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # remove salt-noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_open)

            K_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))     # fuse close fragments
            mask = cv2.dilate(mask, K_dil, iterations=1)

            boxes = components_to_bboxes(mask, min_area=cfg["box"]["min_area"], ignore_border=True)
            # Run NMS
            scores = get_box_scores(boxes, hm_z, mode=cfg["nms"]["mode"])
            score_thr = cfg["nms"]["score_thr"]  # confidence threshold
            nms_thr   = cfg["nms"]["thr"]
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
            if len(indices) > 0:
                indices = indices.flatten()
                boxes = [boxes[i] for i in indices]

            boxes = expand_boxes(boxes, H, W, expand_ratio=cfg["box"]["expand"])  # 5% padding; set 0.0 to disable
            boxes = remove_nested_boxes(boxes, tolerance=0.7)
            
            boxes_vis = draw_boxes(overlay, boxes, color=(0, 0, 255), thickness=2)

            stem = Path(p).stem
            defect = Path(p).parent.name

            for bi, (x,y,w,h) in enumerate(boxes):
                x0, y0, x1, y1 = pad_box(x,y,w,h,H,W,pad_ratio=0.05)
                crop = img[y0:y1, x0:x1]
                cv2.imwrite(os.path.join(out["crops"], f"{defect}_{stem}_box{bi}.png"), crop)
            
            cv2.imwrite(os.path.join(out["images"], f"{defect}_{stem}_orig.png"), img)
            cv2.imwrite(os.path.join(out["images"], f"{defect}_{stem}_overlay.png"), overlay)
            cv2.imwrite(os.path.join(out["images"], f"{defect}_{stem}_mask.png"), mask)
            cv2.imwrite(os.path.join(out["images"], f"{defect}_{stem}_boxes.png"), boxes_vis)
            print(f"images processed: [{idx + k + 1}/{total}]", end="\r")
        idx += bs
                   
    print("\nCropping images completed.",end="\n")

def train_triplet(model , train_loader, val_loader, cfg, out):
    print("Starting triplet learning...")
    CATEGORY = cfg["dataset"]["category"]
    TOTAL_EPOCHS = int(cfg["triplet"]["epochs"])
    MARGIN = float(cfg["triplet"]["margin"])
    LR = float(cfg["triplet"]["lr"])
    WGT_DECAY = float(["triplet"]["weight_decay"])
    TRIPLETPATH = out["base"]["triplet"]
    
    model.train()
    min_err = 10000 # Stores the best validation error so far.
    best_model = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WGT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a,b: 1 - F.cosine_similarity(a,b),
        margin=MARGIN)
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss, total_triplets, contrib_batches = 0.0, 0, 0
        
        # ---- Training START ----
        for x, y, _, _ in train_loader:
            # 1. Move data to GPU
            x = x.cuda()
            y = y.cuda()
            
            # 2. Forward pass
            z = model(x) # shape [B, D]

            # 3. Mine triplets from the batch
            a, p, n = mine_batch_hard(z.detach(), y, MARGIN)
            if len(a) == 0:
                continue # No valid triplets in the batch
            
            # 4. Compute triplet loss on the mined triplets
            loss = triplet_loss(z[a], z[p], z[n])
            
            # 5. Backpropagation and optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # 6. Track stats
            total_loss += loss.item()
            total_triplets += len(a)
            contrib_batches += 1
        
        sched.step()
        avg_loss = total_loss / max(1, contrib_batches)
        print(f"Epoch[{epoch+1}/{TOTAL_EPOCHS}] - train_loss={avg_loss:.4f} | triplets={total_triplets}")
        # ---- Training END ----
        
        # ---- Validation (no grad) START ----
        model.eval()
        val_total_loss, val_total_triplets, val_contrib_batches = 0.0, 0, 0
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                z = model(x)
                z = F.normalize(z, p=2, dim=1)

                a, p, n = mine_batch_hard(z, y, MARGIN)
                if len(a) == 0:
                    continue

                loss = triplet_loss(z[a], z[p], z[n])
                val_total_loss += loss.item()
                val_total_triplets += len(a)
                val_contrib_batches += 1

        # average over number of *batches that produced triplets*
        val_avg_loss = val_total_loss / max(1, val_contrib_batches)
        print(f"val_loss={val_avg_loss:.4f} | val_triplets={val_total_triplets}")
        # ---- Validation END ----
        
        # ---- Save best model ----
        if val_total_triplets > 0 and val_avg_loss < min_err:
            min_err = val_avg_loss
            best_model = {
                'category': CATEGORY,
                "state_dict": model.state_dict()
            }
            torch.save(best_model, os.path.join(TRIPLETPATH, 'triplet_best.pth'))
            
        print(f"Epoch[{epoch+1}/{TOTAL_EPOCHS}] - train_loss={avg_loss:.4f} - triplets={total_triplets} - val_loss={val_avg_loss:.4f} - triplets={val_total_triplets}", end='\r')

    print("Triplet learning completed.")



if __name__ == "__main__":
    main()