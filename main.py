import argparse
import copy
import os
import cv2
import numpy as np
from pathlib import Path

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from evaluate import evaluate
from sklearn.model_selection import train_test_split
from glob import glob

from models.teacher import ResNet18_MS3
from data.mvtec_dataset import MVTecDataset
from data.data_utils import load_ground_truth

from config.config import load_args, load_config

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

    if args.mode == 'train':
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
        apply_threshold(test_err_map, test_loader, args)
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

def apply_threshold(loss_map, loader, args):
    print("Starting to apply threshold and save heatmaps...")
    out_dir = f"outputs/heatmaps/{args.category}"
    os.makedirs(out_dir, exist_ok=True)
    
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
            stem = Path(p).stem
            cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_orig.png"), img)
    print("Finished saving heatmaps.")

if __name__ == "__main__":
    main()