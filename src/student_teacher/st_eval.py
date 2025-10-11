import argparse

from data.datasets import MVTecDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from src.student_teacher.teacher import ResNet18_MS3


def main():
    args = load_args()

    teacher = ResNet18_MS3(pretrained=True)
    student = ResNet18_MS3(pretrained=False)

    st_saved_dict = torch.load(args.st_path)
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



@torch.no_grad()
def get_error_map(teacher, student, loader_or_batch):
    """
    Compute anomaly score maps at 64x64 for either:
      - a full DataLoader that yields (paths, images)
      - a single batch (paths, images)

    Returns: np.ndarray of shape [N, 64, 64]
    """
    teacher.eval()
    student.eval()

    # Figure out what we got (full loader or a single batch)
    if hasattr(loader_or_batch, "dataset"):
        iterable = loader_or_batch
        N = len(loader_or_batch.dataset)
    elif isinstance(loader_or_batch, (list, tuple)) and len(loader_or_batch) == 2:
        paths, x = loader_or_batch
        iterable = [loader_or_batch]
        N = x.size(0)
    elif isinstance(loader_or_batch, list) and loader_or_batch and len(loader_or_batch[0]) == 2:
        iterable = loader_or_batch
        N = sum(batch[1].size(0) for batch in loader_or_batch)
    else:
        raise ValueError("get_error_map expects a DataLoader or (paths, images) batch.")

    loss_map = np.zeros((N, 64, 64), dtype=np.float32)
    i = 0

    for paths, batch_img in iterable:
        batch_img = batch_img.cuda(non_blocking=True)

        # forward
        t_feat = teacher(batch_img)
        s_feat = student(batch_img)

        # aggregate per-level mismatch into a 64x64 map via product
        score_map = 1.0
        for j in range(len(t_feat)):
            t = F.normalize(t_feat[j], dim=1)
            s = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t - s) ** 2, dim=1, keepdim=True)                   # [B,1,h,w]
            sm = F.interpolate(sm, size=(64, 64), mode='bilinear', align_corners=False)
            score_map = score_map * sm                                          # product aggregate

        B = batch_img.size(0)
        loss_map[i:i+B] = score_map.squeeze(1).detach().cpu().numpy()
        i += B

    return loss_map

def load_datasets(img_path, transform):
    print("Loading dataset...")
    # image_list = sorted(glob(os.path.join(args.dataset, args.category, 'test', '*', '*.png')))
    image_list = [img_path]
    dataset = MVTecDataset(image_list, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    print("Loading dataset completed.")
    return loader

def load_calibration_stats(cali_path):
    path = f'{cali_path}/calibration_stats'
    try:
        stats = np.load(path)
        mean = stats['mean']
        std = stats['std']
        print(f"Loaded calibration stats: mean={mean}, std={std}")
        return mean, std
    except Exception as e:
        print(f"Error loading calibration stats from {path}: {e}")
        exit(1)   

def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--img_path", required=True, type=str, help="Path to the image")
    p.add_argument("--category", required=True, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--st_path", required=True, type=str, help="Path to the student model folder")


    args = p.parse_args()
    return args

if __name__ == 'main':
    main()