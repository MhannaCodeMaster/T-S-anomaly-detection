import os
import copy
from types import SimpleNamespace

import torch
from torchvision import transforms
import numpy as np

from src.student_teacher.teacher import ResNet18_MS3
from src.data.data_utils import *

from src.utils.utils import *
from src.paths import get_paths

def main():
    print("Student model training started...")
    args = load_args()
    cfg = load_config()
    cfg = override_config(cfg, args)
    paths = get_paths(cfg.category, 'student')
    
    save_config(cfg, paths.root)

    # Checking if device support cuda.
    if not torch.cuda.is_available():
        print("CUDA is not available. Training requires a CUDA-enabled GPU.")

    print("Training on category: ", cfg.category)
    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)

    teacher.cuda()
    student.cuda()
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # train student model
    train_loader, val_loader = load_st_train_datasets(transform, cfg)
    best_student = train_student(teacher, student, train_loader, val_loader, cfg, paths)
    # Compute calibration stats on training
    mean, std = compute_train_calibration_stats(teacher, best_student, train_loader, paths, device="cuda")
    
    print("Student model training Completed.")


def train_student(teacher, student, train_loader, val_loader, cfg, paths):
    print("Student training started...")
    EPOCHS = int(cfg.epochs)
    LR = float(cfg.lr)
    MOMENTUM = float(cfg.momentum)
    WGT_DECAY = float(cfg.weight_decay)
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
            save_name = os.path.join(paths.checkpoint)
            dir_name = os.path.dirname(save_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            state_dict = {
                'category': cfg.category,
                'state_dict': student.state_dict()
            }
            torch.save(state_dict, save_name)
            best_student = copy.deepcopy(student)
        print(f"Epoch: [{epoch+1}/{cfg.epochs}] - Avg training loss: {avg_train_loss:.6f} - Validation loss: {err_mean.item():.7f}",end="\r")
        
    print("\nStudent training completed.")
    return best_student

@torch.no_grad()
def compute_train_calibration_stats(teacher, student, train_loader, paths, device="cuda"):
    """
    Compute scalar μ, σ over all pixels of anomaly maps on normal training images.
    Saves to npz: {mean: float, std: float}.
    Returns mean and std
    """
    print("Computing training set calibration stats...")
    teacher.eval(); student.eval()
    teacher.to(device); student.to(device)
    
    sum1 = torch.zeros((), dtype=torch.float64, device=device)
    sum2 = torch.zeros((), dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=device)

    for _, batch_img in train_loader:
        batch_img = batch_img.to(device, non_blocking=True)
        t_feat = teacher(batch_img)
        s_feat = student(batch_img)

        # feature mismatch -> (N,1,h,w) per level -> upsample to a common 64x64 -> sum across levels
        score = None
        for t, s in zip(t_feat, s_feat):
            t = F.normalize(t, dim=1)
            s = F.normalize(s, dim=1)
            # per-pixel channel MSE
            diff = (t - s).pow(2).sum(dim=1, keepdim=True)   # (N,1,h,w)
            diff64 = F.interpolate(diff, size=(64, 64), mode="bilinear", align_corners=False)
            score = diff64 if score is None else (score * diff64)

        flat = score.reshape(-1).to(dtype=torch.float64)
        sum1  += flat.sum()
        sum2 += (flat * flat).sum()
        count += flat.numel()

    mean = (sum1 / count).item()
    var  = (sum2 / count).item() - mean * mean
    std  = float(np.sqrt(max(var, 1e-12)))

    file_path = os.path.join(paths.calibration)
    np.savez(file_path, mean=float(mean), std=float(std))
    print(f"Calibration saved μ={mean:.6g}, σ={std:.6g} -> {file_path}")
    return mean, std

def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Path to mvtec dataset root directory")
    
    #----- Optional args -----#
    p.add_argument("--category", required=False, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--epochs", type=int, required=False, help="Number of student training epochs")
    p.add_argument("--batch_size", type=int, required=False, help="Batch size for student training")
    p.add_argument("--lr", type=float, required=False, help="Learning rate for student training")
    p.add_argument("--weight_decay", type=float, required=False, help="Weight decay for student training")

    args = p.parse_args()
    return args

def load_config():
    CONF_PATH = "../../conf/student.yaml"
    try:
        with open(CONF_PATH, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                raise
    except FileNotFoundError:
        print(f"Config file not found: {CONF_PATH}")
        raise
    except Exception as e:
        print(f"Unexpected error reading config file: {e}")
        raise

    return dict_to_namespace(config)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d
    
def override_config(cfg, args):
    def recursive_set(obj, key_path, value):
        keys = key_path.split(".")
        cur = obj
        for k in keys[:-1]:
            cur = getattr(cur, k)
        setattr(cur, keys[-1], value)

    for key, value in vars(args).items():
        if value is not None:
            recursive_set(cfg, key, value)
    return cfg


if __name__ == "__main__":
    main()
