import argparse

import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Sampler

from src.data.data_utils import *
from src.triplet.triplet import TripletEmbedder
from src.data.datasets import *

def main():
    args = load_args()
    print("Evaluation triplet model...")
    
    print("Evaluating on category: ", args.category)
    try:
        loader, _, _ = load_crops(args)
        model = TripletEmbedder(pretrained=False)
        model.cuda()
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
        emd, label = extract_embeddings(model, loader)
        plot_tsne(emd, label, args)
    except Exception as e:
        print("Error has occured: ", e)
    print("Evaluation completed...")
    
def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Path to the folder crops with ok/not ok dataset root directory")
    p.add_argument("--mainfest_path", required=True, type=str, help="Path to the folder containing parents.csv")
    p.add_argument("--model_path", required=True, type=str, help="Path to the trained triplet model")
    
    #----- Optional args -----#
    p.add_argument("--category", required=False, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--batch_size", type=int, required=False, help="Batch size for triplet training")
    p.add_argument("--tsne_components", required=False, type=str, help="T-SNE components (2: 2D or 3: 3D) ")

    args = p.parse_args()
    return args

def load_crops(args, img_size=224, num_workers=4, pin_memory=True):
    """
    Rebuild  crops using saved parents.csv under `paths.root`,
    then return (loader, dataset, df) for evaluation (e.g., t-SNE).
    """
    dataset_root = args.dataset # path to your dataset root
    crops_dir = os.path.join(dataset_root, "crops")

    # 1) Read manifests
    ok_df    = pd.read_csv(os.path.join(crops_dir, "ok_manifest.csv"))
    notok_df = pd.read_csv(os.path.join(crops_dir, "not_ok_manifest.csv"))

    # 2) Read validation parent IDs (saved during your split)
    parents_path = os.path.join(args.mainfest_path, "train_parents.csv")
    parents = set(pd.read_csv(parents_path)["parent_id"])

    # 3) Filter to validation crops only
    ok_df    = ok_df[ok_df["parent_id"].isin(parents)].copy()
    notok_df = notok_df[notok_df["parent_id"].isin(parents)].copy()
    ok_df["label"] = "ok"
    notok_df["label"] = "not_ok"
    df = pd.concat([ok_df, notok_df], ignore_index=True)

    # 4) Transforms (match your training/eval normalization)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # 5) Build datasets exactly like your loader function does
    #    (keep per-class datasets if your PatchDataset expects class via arg)
    ok_ds    = PatchDataset(ok_df,    tf, "ok",      crops_root=dataset_root)
    notok_ds = PatchDataset(notok_df, tf, "not_ok",  crops_root=dataset_root)
    dataset  = torch.utils.data.ConcatDataset([ok_ds, notok_ds])
    
    len_ok = len(ok_ds)
    len_ng = len(notok_ds)
    sampler = StratifiedTwoClassBatchSampler(len_ok=len_ok, len_ng=len_ng, batch_size=args.batch_size, drop_last=False)
    
    # 6) DataLoader
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=True)

    print(f"[train] parents={len(parents)} | crops={len(df)} | ok={len(ok_df)} | not_ok={len(notok_df)}")
    return loader, dataset, df       
       
@torch.no_grad()
def extract_embeddings(model, loader, device="cuda"):
    model.eval()
    embs, labels = [], []
    for x, y, _, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)                     # already L2-normalized (B, D)
        embs.append(z.cpu())
        labels.append(y.cpu())
    embs = torch.cat(embs, dim=0).numpy()     # (N, D)
    labels = torch.cat(labels, dim=0).numpy() # (N,)
    return embs, labels

def plot_tsne(embs, labels, cfg):
    dim = int(cfg.tsne_components)
    # t-SNE on normalized embeddings; Euclidean ~ cosine on unit sphere
    tsne = TSNE(
        n_components=dim,
        init="pca",
        perplexity=30,
        learning_rate="auto",
        metric="euclidean",
        random_state=42,
        n_iter=1000,
        verbose=1,
    )
    Z = tsne.fit_transform(embs)  # (N, 2)

    if dim == 2:
        plt.figure(figsize=(7, 6), dpi=120)
        for u in np.unique(labels):
            m = labels == u
            plt.scatter(Z[m, 0], Z[m, 1], s=14, alpha=0.85, label=str(u))
        plt.title("t-SNE of Triplet Embeddings (val)")
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
        plt.legend(title="Label", frameon=True)
        plt.tight_layout()
        plt.savefig('TSNE_triplet.png', bbox_inches="tight")
        plt.close()
    elif dim == 3:
        fig = plt.figure(figsize=(8, 6), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        for u in np.unique(labels):
            m = labels == u
            ax.scatter(Z[m, 0], Z[m, 1], Z[m, 2], s=14, alpha=0.85, label=str(u))
        ax.set_title("t-SNE (3D) of Triplet Embeddings (val)")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_zlabel("t-SNE 3")
        ax.legend(title="Label")
        fig.tight_layout()
        fig.savefig('TSNE_triplet.png', bbox_inches="tight")
        plt.close(fig)
    else:
        raise ValueError("dim must be 2 or 3")


class StratifiedTwoClassBatchSampler(Sampler):
    """Yields indices for batches containing half OK and half NOT_OK."""
    def __init__(self, len_ok, len_ng, batch_size, drop_last=False):
        assert batch_size % 2 == 0, "Use even batch_size"
        self.len_ok, self.len_ng = len_ok, len_ng
        self.bs = batch_size
        self.drop_last = drop_last
        self.ok_idx = list(range(0, len_ok))
        self.ng_idx = list(range(len_ok, len_ok+len_ng))

    def __iter__(self):
        ok = self.ok_idx[:]
        ng = self.ng_idx[:]
        random.shuffle(ok); random.shuffle(ng)
        i = j = 0
        while i + self.bs//2 <= len(ok) and j + self.bs//2 <= len(ng):
            batch = ok[i:i+self.bs//2] + ng[j:j+self.bs//2]
            random.shuffle(batch)
            yield batch
            i += self.bs//2; j += self.bs//2
        if not self.drop_last:
            # handle remainders (optional: top-up with random)
            pass

    def __len__(self):
        return min(self.len_ok, self.len_ng) * 2 // self.bs
