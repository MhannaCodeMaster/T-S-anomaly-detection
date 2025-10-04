import os
from types import SimpleNamespace

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.triplet.triplet import *
from src.data.data_utils import *
from src.data.datasets import *

from conf.config import *
from src.utils.utils import *
from src.paths import get_paths

def main():
    print("Triplet learning started...")
    args = load_args()
    cfg = load_config(args)
    cfg = override_config(cfg, args)
    paths = get_paths(cfg.category, 'triplet')
    print("Training on category: ", cfg.category)

    triplet = TripletEmbedder(pretrained=True)
    triplet.cuda()
    train_loader, val_loader = load_tl_training_datasets(cfg, paths)
    train_triplet(triplet, train_loader, val_loader, cfg, paths)
   
   
def train_triplet(model , train_loader, val_loader, cfg, paths):
    print("Starting triplet learning...")

    #----------- CONFIG -----------#
    CATEGORY     = cfg.category
    TOTAL_EPOCHS = int(cfg.epochs)
    BATCH_SIZE   = int(cfg.batch_size)
    MARGIN       = float(cfg.margin)
    LR           = float(cfg.lr)
    WGT_DECAY    = float(cfg.weight_decay)
    TRIPLETPATH  = paths.checkpoint

    # adaptive-margin controller
    M_MIN, M_MAX          = 0.30, 0.80
    STEP                  = 0.05
    EMA_BETA              = 0.8           # smoothing for val_active
    TARGET_LOW, TARGET_HIGH = 0.30, 0.40  # we want ~30–40% active on val
    COOLDOWN_EPOCHS       = 3
    cooldown              = 0
    ema_val_active        = None

    # early stopping
    ES_PATIENCE  = getattr(cfg, "es_patience", 20)       # epochs with no meaningful improvement
    ES_MIN_DELTA = getattr(cfg, "es_min_delta", 1e-3)    # minimum improvement in val_loss
    best_val_loss = float("inf")
    es_bad_epochs = 0

    best_model = None
    
    optimizer    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WGT_DECAY)
    sched        = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a,b: 1 - F.cosine_similarity(a,b),
        margin=MARGIN)
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_train_loss, total_train_triplets = 0.0, 0
        train_active, train_active_total, train_batches = 0, 0, 0

        #---------------- TRAIN START ----------------# 
        for x, y, _, _ in train_loader:
            x = x.cuda() # tensor of shape [batch_size, channels, height, width]
            y = y.cuda() # tensor of shape [batch_size]
            z = model(x) # tensor of shape [batch_size, embed_dim]
            
            # Mining triplets from the batch for the triplet loss algo
            # we take one anchor, one positive, and one negative
            # a, p, n = mine_batch(z.detach(), y, MARGIN)
            # a, p, n = mine_triplets(z.detach(), y)
            a, p, n = mine_triplets_v1(z.detach(), y, margin=MARGIN)
            if len(a) == 0:
                continue
            
            loss = triplet_loss(z[a], z[p], z[n])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # act_pct = mining_stats(z.detach(), y, MARGIN)
            # train_active_sum += act_pct
            # train_batches += 1
            
            total_train_triplets += len(a)
            act_pct, total = mining_active(z.detach(), y, MARGIN)
            train_active += act_pct
            train_active_total += total

        sched.step()
        #---- Training Metrics ----#
        epoch_active_pct = 100.0 * train_active / train_active_total if train_active_total > 0 else 0.0
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Training stats - Epoch {epoch+1} -  avg train loss: {avg_train_loss:.4f} - train active={epoch_active_pct}")
        #---------------- TRAIN END ----------------# 
        
        #---------------- VAL START ----------------#
        val_total_loss, val_total_triplets, val_contrib_batches = 0.0, 0, 0
        val_active, val_active_total = 0,0
        avg_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y, _, _ in val_loader:
                x = x.cuda()
                y = y.cuda()
                z = model(x)
                
                # a, p, n = mine_batch(z.detach(), y, MARGIN)
                # a, p, n = mine_triplets(z.detach(), y)
                a, p, n = mine_triplets_v1(z.detach(), y, margin=MARGIN)
                if len(a) == 0:
                    continue
                
                loss = triplet_loss(z[a], z[p], z[n])
                
                val_total_triplets += len(a)
                val_total_loss += loss.item()
                act_pct, total = mining_active(z.detach(), y, MARGIN)
                val_active += act_pct
                val_active_total += total
            
        #---- Val Metrics ----#
        epoch_active_pct = 100.0 * val_active / val_active_total if val_active_total > 0 else 0.0
        avg_val_loss = val_total_loss / len(val_loader)
        print(f"Validation stats - Epoch {epoch+1} -  avg val loss: {avg_val_loss:.4f} - val active={epoch_active_pct}")
        #---------------- VAL END ----------------#

        #----- best/early stop -----
        improved = (best_val_loss - avg_val_loss) > ES_MIN_DELTA
        if improved:
            best_val_loss = avg_val_loss
            es_bad_epochs = 0
            torch.save({'category': CATEGORY, 'state_dict': model.state_dict()}, os.path.join(TRIPLETPATH))
        else:
            es_bad_epochs += 1

        # ----- early stopping -----
        if es_bad_epochs >= ES_PATIENCE:
            print(f"Early stopping: no val_loss improvement > {ES_MIN_DELTA} for {ES_PATIENCE} epochs.")
            break
        
        # # ----- adaptive margin (convert to FRACTION here) -----
        # if ema_val_active is None:
        #     ema_val_active = val_active_frac
        # else:
        #     ema_val_active = EMA_BETA * ema_val_active + (1 - EMA_BETA) * val_active_frac

        # if cooldown == 0:
        #     if ema_val_active < TARGET_LOW:
        #         MARGIN = min(M_MAX, MARGIN + STEP)
        #         cooldown = COOLDOWN_EPOCHS
        #         triplet_loss.margin = MARGIN
        #     elif ema_val_active > TARGET_HIGH:
        #         MARGIN = max(M_MIN, MARGIN - STEP)
        #         cooldown = COOLDOWN_EPOCHS
        #         triplet_loss.margin = MARGIN
        # else:
        #     cooldown -= 1

    print("Triplet learning completed.")

@torch.no_grad()
def mine_triplets(z, y):
    """
    Random mining inside the batch.
    Inputs:
      z: [B, D]
      y: [B]
    Returns:
      a, p, n : Long tensors of indices on y.device
    """
    B = y.size(0)
    device = y.device
    a_idx, p_idx, n_idx = [], [], []

    for a in range(B):
        same = (y == y[a])
        diff = ~same
        same[a] = False  # don't pick the anchor as positive

        if not same.any() or not diff.any():
            continue

        pos_idx = torch.where(same)[0]
        p = pos_idx[torch.randint(len(pos_idx), (1,)).item()]

        neg_idx = torch.where(diff)[0]
        n = neg_idx[torch.randint(len(neg_idx), (1,)).item()]

        a_idx.append(a); p_idx.append(int(p)); n_idx.append(int(n))

    if len(a_idx) == 0:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device))
    return (torch.tensor(a_idx, dtype=torch.long, device=device),
            torch.tensor(p_idx, dtype=torch.long, device=device),
            torch.tensor(n_idx, dtype=torch.long, device=device))

@torch.no_grad()
def mine_triplets_v1(
    z: torch.Tensor,
    y: torch.Tensor,
    mode: str = "semi-hard",
    margin: float = 0.5,
    parent_ids: list | None = None,
):
    """
    Mine triplets (a, p, n) from a batch of embeddings.

    Args:
        z: [B, D] L2-normalized embeddings (unit norm -> cosine distance works).
        y: [B] integer labels.
        mode: "random" | "semi-hard" | "hard".
              - random   : random positive, random negative from other classes
              - semi-hard: hardest positive; negative with d_ap < d_an < d_ap+margin (fallback to hard-neg)
              - hard     : hardest positive; hardest negative (closest negative)
        margin: margin used for semi-hard selection.
        parent_ids: optional list of len B; if provided, negatives sharing the
                    same parent as the anchor are excluded (avoid false negatives).

    Returns:
        a, p, n: Long tensors of indices on the same device as y.
    """
    device = y.device
    B = y.size(0)
    # Cosine distance since z is unit-normalized: d = 1 - cos_sim in [0, 2]
    D = 1.0 - z @ z.t()  # [B,B]

    # Optional: map parent ids to ints for quick comparison on device
    pid = None
    if parent_ids is not None:
        uniq = {p: i for i, p in enumerate(parent_ids)}
        pid = torch.tensor([uniq[p] for p in parent_ids], device=device)

    aL, pL, nL = [], [], []

    for a in range(B):
        same = (y == y[a]).clone()
        diff = ~same
        same[a] = False  # exclude the anchor itself as a positive

        # Exclude negatives from same parent if requested
        if pid is not None:
            diff = diff & (pid != pid[a])

        if not same.any() or not diff.any():
            continue

        # ---------- choose positive ----------
        if mode == "random":
            pos_idx = torch.where(same)[0]
            p = pos_idx[torch.randint(len(pos_idx), (1,)).item()]
            d_ap = D[a, p].item()
        else:
            # hardest positive (max distance among positives)
            pos_dists = D[a][same]                           # [P]
            p_rel = torch.argmax(pos_dists)                  # idx within mask
            p = torch.arange(B, device=device)[same][p_rel]  # absolute index
            d_ap = float(D[a, p])

        # ---------- choose negative ----------
        if mode == "random":
            neg_idx = torch.where(diff)[0]
            n = neg_idx[torch.randint(len(neg_idx), (1,)).item()]

        elif mode == "hard":
            # hardest negative = closest negative (min distance)
            n_rel = torch.argmin(D[a][diff])
            n = torch.arange(B, device=device)[diff][n_rel]

        elif mode == "semi-hard":
            # window: d_ap < d_an < d_ap + margin
            window = (D[a] > d_ap) & (D[a] < d_ap + margin) & diff
            if window.any():
                # pick the closest in the window (hardest among semi-hard)
                n_rel = torch.argmin(D[a][window])
                n = torch.arange(B, device=device)[window][n_rel]
            else:
                # fallback to hardest negative
                n_rel = torch.argmin(D[a][diff])
                n = torch.arange(B, device=device)[diff][n_rel]
        else:
            raise ValueError(f"Unknown mining mode: {mode}")

        aL.append(a); pL.append(int(p)); nL.append(int(n))

    if not aL:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, empty

    return (
        torch.tensor(aL, dtype=torch.long, device=device),
        torch.tensor(pL, dtype=torch.long, device=device),
        torch.tensor(nL, dtype=torch.long, device=device),
    )

   
def mining_active(z, y, margin):
    B = y.size(0)
    D = 1.0 - z @ z.t()
    act_count, total = 0, 0
    
    for a in range(B):
        same = (y == y[a])
        diff = ~same
        if not same.any() or not diff.any():
            continue
        
        for p in torch.where(same)[0]:
            d_ap = D[a, p].item()
            for n in torch.where(diff)[0]:
                d_an = D[a, n].item()
                total += 1
                if d_ap + margin > d_an:  # active
                    act_count += 1
                    
    if total == 0:
        return 0.0
    return act_count, total
 
def load_tl_training_datasets(cfg, paths):
    print("Loading triplet training datasets...")
    BATCH_SIZE = cfg.batch_size
    SEED = 123
    VAL_PERC = 0.3
    
    print("Reading manifests...")
    ok_df = pd.read_csv(os.path.join(cfg.dataset, 'crops', 'ok_manifest.csv'))
    notok_df = pd.read_csv(os.path.join(cfg.dataset, 'crops', 'not_ok_manifest.csv'))
    
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

    train_parents_df.to_csv(os.path.join(paths.root, "train_parents.csv"), index=False)
    val_parents_df.to_csv(os.path.join(paths.root,"val_parents.csv"), index=False)
    
    print("Train parents:", len(train_parents))
    print("Val parents:", len(val_parents))
    
    # Train/Val splits
    train_ok_df = ok_df[ok_df['parent_id'].isin(train_parents)]
    train_notok_df = notok_df[notok_df['parent_id'].isin(train_parents)]
    val_ok_df = ok_df[ok_df['parent_id'].isin(val_parents)]
    val_notok_df = notok_df[notok_df['parent_id'].isin(val_parents)]
    print("Train OK:", len(train_ok_df), "Train NOT_OK:", len(train_notok_df))
    print("Val   OK:", len(val_ok_df),   "Val   NOT_OK:", len(val_notok_df))
    
    img_size = 224
    
    train_ok_tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.20, hue=0.02),
        transforms.RandomAutocontrast(p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        # --- Zoom IN: crop a smaller area & resize back to 224x224
        transforms.RandomApply([
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.70, 1.00),          # 70%..100% of area -> zoom-in
                ratio=(0.90, 1.10),          # keep aspect near 1:1 for cables
                interpolation=InterpolationMode.BILINEAR
            )
        ], p=0.50),

        # --- Zoom OUT + slight shifts/rotations (center preserved)
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=5,                   # small rotation
                translate=(0.05, 0.05),      # up to ±5% shift
                scale=(0.85, 1.25),          # <1 = zoom-in-ish; >1 = zoom-out
                shear=None,
                interpolation=InterpolationMode.BILINEAR,
                fill=0                        # or tuple of means if you prefer
            )
        ], p=0.80),

        # (Optional) Tiny perspective jitter if cables can bend
        transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.15, p=1.0)], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

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
    train_ok_ds    = PatchDataset(train_ok_df, train_ok_tf, "ok", crops_root=cfg.dataset)
    train_notok_ds = PatchDataset(train_notok_df, train_tf, "not_ok", crops_root=cfg.dataset)
    val_ok_ds      = PatchDataset(val_ok_df, val_tf, "ok", crops_root=cfg.dataset)
    val_notok_ds   = PatchDataset(val_notok_df, val_tf, "not_ok", crops_root=cfg.dataset)
    
    train_dataset = torch.utils.data.ConcatDataset([train_ok_ds, train_notok_ds])
    val_dataset   = torch.utils.data.ConcatDataset([val_ok_ds, val_notok_ds])
    len_ok = len(train_ok_ds)
    len_ng = len(train_notok_ds)
    train_sampler = StratifiedTwoClassBatchSampler(len_ok=len_ok, len_ng=len_ng, batch_size=BATCH_SIZE, drop_last=True)   

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                            num_workers=4, pin_memory=True)
    print("Train dataset loaded")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    print("Validation dataset loaded")
    print("Triplet datasets loading completed.")
    return train_loader, val_loader

def load_args():
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--dataset", required=True, type=str, help="Path to the folder crops with ok/not ok dataset root directory")
    
    #----- Optional args -----#
    p.add_argument("--category", required=False, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--epochs", type=int, required=False, help="Number of triplet training epochs")
    p.add_argument("--batch_size", type=int, required=False, help="Batch size for triplet training")
    p.add_argument("--lr", type=float, required=False, help="Learning rate for triplet training")
    p.add_argument("--weight_decay", type=float, required=False, help="Weight decay for triplet training")
    p.add_argument("--momentum", type=float, required=False, help="Momentum for triplet training")
    p.add_argument("--config", required=False, type=str, help="Config path")
    p.add_argument("--margin", required=False, type=str, help="Triplet margin")

    args = p.parse_args()
    return args

def load_config(args):
    CONF_PATH = args.config
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

import torch.nn.functional as F

@torch.no_grad()
def mining_stats(emb, labels, margin: float, kth: int = 5):
    """
    % anchors violating (hardest positive vs kth-nearest negative) in cosine space.
    emb: [N,D], labels: [N], margin in cosine distance space.
    """
    # L2-normalize for stable cosine geometry
    emb = torch.nn.functional.normalize(emb, dim=1)

    # cosine distance in [0,2]
    D = 1.0 - emb @ emb.t()

    N = labels.numel()
    device = labels.device
    same = labels[:, None].eq(labels[None, :])
    eye  = torch.eye(N, dtype=torch.bool, device=device)

    pos_mask = same & (~eye)
    neg_mask = ~same

    # hardest positive
    pos_d = D.clone()
    pos_d[~pos_mask] = -1.0
    hardest_pos = pos_d.max(dim=1).values  # [-1,2]

    # kth-nearest negative (avoid the single closest outlier)
    neg_d = D.clone()
    neg_d[~neg_mask] = float('inf')
    n_negs = neg_mask.sum(1)

    valid = (hardest_pos > -0.5) & (n_negs >= kth)
    if not valid.any():
        return float("nan")

    # kth smallest negative distance per valid row
    kth_vals = torch.topk(-neg_d[valid], kth, dim=1).values[:, -1].neg()

    delta   = hardest_pos[valid] - kth_vals + margin
    active  = (delta > 0).float()
    return active.mean().item() * 100.0


# @torch.no_grad()
# def mining_stats(emb, labels, margin: float):
#     """Compute % of anchors violating the margin (active triplets)."""
#     D = 1.0 - emb @ emb.t()  # cosine distance in [0,2]

#     same = labels[:, None].eq(labels[None, :])
#     eye = torch.eye(len(labels), device=labels.device, dtype=torch.bool)

#     pos_mask = same & (~eye)
#     neg_mask = ~same

#     # hardest positive and negative per anchor
#     pos = (D * pos_mask.float())
#     pos[pos_mask == 0] = -1.0
#     hardest_pos = pos.max(dim=1).values

#     Dn = D.clone()
#     Dn[neg_mask == 0] = 1e9
#     hardest_neg = Dn.min(dim=1).values

#     delta = hardest_pos - hardest_neg + margin
#     active = (delta > 0).float()

#     valid = (hardest_pos > -0.5) & (hardest_neg < 1e8)
#     if valid.any():
#         return active[valid].mean().item() * 100.0  # percentage
#     else:
#         return float("nan")
    
if __name__ == "__main__":
    main()
