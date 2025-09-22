import argparse

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.data.data_utils import *
from src.triplet.triplet import TripletEmbedder


def main():
    print("Evaluation triplet model...")
    args = load_args()
    print("Evaluating on category: ", args.category)
    
    try:
        val_loader = load_val_crops(args)
        model = TripletEmbedder(pretrained=False)
        model.cuda()
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
        emd, label = extract_embeddings(model, val_loader)
        plot_tsne(emd, label)
    except Exception as e:
        print("Error has occured: ", e)
    print("Evaluation completed...")
    
def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Path to the folder crops with ok/not ok dataset root directory")
    
    #----- Optional args -----#
    p.add_argument("--category", required=False, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--batch_size", type=int, required=False, help="Batch size for triplet training")
    p.add_argument("--val_mainfest_path", required=True, type=str, help="Path to the folder containing val_parents.csv")
    p.add_argument("--model_path", required=True, type=str, help="Path to the trained triplet model")
    

    args = p.parse_args()
    return args

def load_val_crops(args, img_size=224, num_workers=4, pin_memory=True):
    """
    Rebuild validation crops using saved val_parents.csv under `paths.root`,
    then return (val_loader, val_dataset, val_df) for evaluation (e.g., t-SNE).
    """
    dataset_root = args.dataset # path to your dataset root
    crops_dir = os.path.join(dataset_root, "crops")

    # 1) Read manifests
    ok_df    = pd.read_csv(os.path.join(crops_dir, "ok_manifest.csv"))
    notok_df = pd.read_csv(os.path.join(crops_dir, "not_ok_manifest.csv"))

    # 2) Read validation parent IDs (saved during your split)
    val_parents_path = os.path.join(args.val_mainfest_path, "val_parents.csv")
    val_parents = set(pd.read_csv(val_parents_path)["parent_id"])

    # 3) Filter to validation crops only
    val_ok_df    = ok_df[ok_df["parent_id"].isin(val_parents)].copy()
    val_notok_df = notok_df[notok_df["parent_id"].isin(val_parents)].copy()
    val_ok_df["label"] = "ok"
    val_notok_df["label"] = "not_ok"
    val_df = pd.concat([val_ok_df, val_notok_df], ignore_index=True)

    # 4) Transforms (match your training/eval normalization)
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    # 5) Build datasets exactly like your loader function does
    #    (keep per-class datasets if your PatchDataset expects class via arg)
    val_ok_ds    = PatchDataset(val_ok_df,    val_tf, "ok",      crops_root=dataset_root)
    val_notok_ds = PatchDataset(val_notok_df, val_tf, "not_ok",  crops_root=dataset_root)
    val_dataset  = torch.utils.data.ConcatDataset([val_ok_ds, val_notok_ds])

    # 6) DataLoader
    bs =  args.batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"[val] parents={len(val_parents)} | crops={len(val_df)} | ok={len(val_ok_df)} | not_ok={len(val_notok_df)}")
    return val_loader, val_dataset, val_df       
       
@torch.no_grad()
def extract_embeddings(model, loader, device="cuda"):
    model.eval()
    embs, labels = [], []
    for batch in loader:
        # Accept (images, labels) or (paths, images, labels)
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                _, x, y = batch
            else:
                raise ValueError("Unexpected batch format for val loader.")
        else:
            raise ValueError("Val loader must return (x,y) or (path,x,y).")

        x = x.to(device, non_blocking=True)
        z = model(x)                     # already L2-normalized (B, D)
        embs.append(z.cpu())
        labels.append(y.cpu())
    embs = torch.cat(embs, dim=0).numpy()     # (N, D)
    labels = torch.cat(labels, dim=0).numpy() # (N,)
    return embs, labels

def plot_tsne(embs, labels):
    # t-SNE on normalized embeddings; Euclidean ~ cosine on unit sphere
    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=30,
        learning_rate="auto",
        metric="euclidean",
        random_state=42,
        n_iter=1000,
        verbose=1,
    )
    xy = tsne.fit_transform(embs)  # (N, 2)

    # Scatter: color by ground-truth label (let matplotlib pick default colors)
    plt.figure(figsize=(7, 6), dpi=120)
    uniq = np.unique(labels)
    for u in uniq:
        m = labels == u
        plt.scatter(xy[m, 0], xy[m, 1], s=14, alpha=0.85, label=str(u))
    plt.title("t-SNE of Triplet Embeddings (val)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Label", loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig("eval/triplet/images", bbox_inches="tight")
    plt.close()