import os
import cv2
import numpy as np
import pandas as pd
import random

from data.datasets import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
import yaml


def load_ground_truth(root, category):
    gt = []
    gt_dir = os.path.join(root, category, 'ground_truth')
    sub_dirs = sorted(os.listdir(gt_dir))
    for sb in sub_dirs:
        for fname in sorted(os.listdir(os.path.join(gt_dir, sb))):
            temp = cv2.imread(os.path.join(gt_dir, sb, fname), cv2.IMREAD_GRAYSCALE)
            temp = cv2.resize(temp, (256, 256)).astype(np.bool)[None, ...]
            gt.append(temp)
    gt = np.concatenate(gt, 0)
    return gt

def load_st_train_datasets(transform, cfg):
    print("Loading T-S datasets...")
    image_list = sorted(glob(os.path.join(cfg["dataset"]["mvtec_path"], cfg["dataset"]["mvtec_category"], 'train', 'good', '*.png')))
    train_image_list, val_image_list = train_test_split(image_list, test_size=0.2, random_state=0)
    train_dataset = MVTecDataset(train_image_list, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg["student"]["batch_size"], shuffle=True, drop_last=False)
    print("Training dataset loaded")
    val_dataset = MVTecDataset(val_image_list, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg["student"]["batch_size"], shuffle=False, drop_last=False)
    print("Validation dataset loaded")
    print("T-S datasets loading completed.")
    return train_loader, val_loader

def load_tl_datasets(cfg, out):
    print("Loading triplet learning datasets...")
    SEED = 123
    VAL_PERC = 0.2
    
    print("Reading manifests...")
    ok_df = pd.read_csv(cfg["dataset"]["ok_manifest"])
    notok_df = pd.read_csv(cfg["dataset"]["notok_manifest"])
    
    # We need to group the patches by their source image
    parents = set(ok_df['parent_id']).union(set(notok_df['parent_id']))
    parents = sorted(parents)
    print(f"Total of parent images: {len(parents)}")
    
    random.seed(SEED)
    random.shuffle(parents)
    cut = max(1, int(len(parents) * VAL_PERC))
    val_parents = set(parents[:cut])
    train_parents = set(parents[cut:])
    
    # Saving parent_id lists
    train_parents_df = pd.DataFrame({"parent_id": list(train_parents)})
    val_parents_df   = pd.DataFrame({"parent_id": list(val_parents)})

    train_parents_df.to_csv(os.path.join(out['base']['crops'], "train_parents.csv"), index=False)
    val_parents_df.to_csv(os.path.join(out['base']['crops'],"val_parents.csv"), index=False)
    
    print("Train parents:", len(train_parents))
    print("Val parents:", len(val_parents))
    
    # Train/Val splits
    train_ok_df = ok_df[ok_df['parent_id'].isin(train_parents)]
    train_notok_df = notok_df[notok_df['parent_id'].isin(train_parents)]
    val_ok_df = ok_df[ok_df['parent_id'].isin(val_parents)]
    val_notok_df = notok_df[notok_df['base_ok_filename'].isin(val_parents)]
    print("Train OK:", len(train_ok_df), "Train NOT_OK:", len(train_notok_df))
    print("Val   OK:", len(val_ok_df),   "Val   NOT_OK:", len(val_notok_df))
    
    img_size = 224
    train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(12, fill=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    
    # Building the datasets
    train_ok_ds    = PatchDataset(train_ok_df, train_tf, "ok", crops_root=cfg['dataset']['crops_root'])
    train_notok_ds = PatchDataset(train_notok_df, train_tf, "not_ok", crops_root=cfg['dataset']['crops_root'])
    val_ok_ds      = PatchDataset(val_ok_df, val_tf, "ok", crops_root=cfg['dataset']['crops_root'])
    val_notok_ds   = PatchDataset(val_notok_df, val_tf, "not_ok", crops_root=cfg['dataset']['crops_root'])
    
    train_dataset = torch.utils.data.ConcatDataset([train_ok_ds, train_notok_ds])
    val_dataset   = torch.utils.data.ConcatDataset([val_ok_ds, val_notok_ds])
    
    BATCH_SIZE = cfg["triplet"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    print("Train dataset loaded")
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    print("Validation dataset loaded")
    print("Triplet datasets loading completed.")
    return train_loader, val_loader

def load_test_datasets(transform, cfg):
    print("Loading test datasets...")
    test_neg_image_list = sorted(glob(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["category"], 'test', 'good', '*.png')))
    test_pos_image_list = set(glob(os.path.join(cfg["dataset"]["root"], cfg["dataset"]["category"], 'test', '*', '*.png'))) - set(test_neg_image_list)
    test_pos_image_list = sorted(list(test_pos_image_list))
    test_neg_dataset = MVTecDataset(test_neg_image_list, transform=transform)
    test_pos_dataset = MVTecDataset(test_pos_image_list, transform=transform)
    test_neg_loader = DataLoader(test_neg_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_pos_loader = DataLoader(test_pos_dataset, batch_size=1, shuffle=False, drop_last=False)
    print("Test datasets loading completed.")
    return test_neg_loader, test_pos_loader

def load_calibration_stats(cali_path):
    try:
        stats = np.load(cali_path)
        mean = stats['mean']
        std = stats['std']
        print(f"Loaded calibration stats: mean={mean}, std={std}")
        return mean, std
    except Exception as e:
        print(f"Error loading calibration stats from {cali_path}: {e}")
        exit(1)

def save_config(cfg, path):
    config_save_path = os.path.join(path, "config.yaml")
    try:
        with open(config_save_path, 'w') as file:
            yaml.dump(cfg, file)
    except Exception as e:
        print(f"Error saving configuration to {config_save_path}: {e}")
        exit(1)

