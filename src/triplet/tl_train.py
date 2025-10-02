import os
from types import SimpleNamespace

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


from src.triplet.triplet import TripletEmbedder
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

    try:
        triplet = TripletEmbedder(pretrained=True)
        triplet.cuda()
        train_loader, val_loader = load_tl_training_datasets(cfg, paths)
        train_triplet(triplet, train_loader, val_loader, cfg, paths)
    except Exception as e:
        print("Error has occured: ", e)
   
def train_triplet(model , train_loader, val_loader, cfg, paths):
    print("Starting triplet learning...")

    #----------- CONFIG -----------#
    CATEGORY     = cfg.category
    TOTAL_EPOCHS = int(cfg.epochs)
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

    model.train()
    best_model = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WGT_DECAY)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    triplet_loss = torch.nn.TripletMarginWithDistanceLoss(
        distance_function=lambda a,b: 1 - F.cosine_similarity(a,b),
        margin=MARGIN
    )
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss, total_triplets, contrib_batches = 0.0, 0, 0

        # --- reset per-epoch stats ---
        train_active_sum, train_batches = 0.0, 0
        val_active_sum,   val_batches   = 0.0, 0

        # ---------- TRAIN ----------
        for x, y, _, _ in train_loader:
            x = x.cuda(); y = y.cuda()
            z = model(x)  # already normalized

            act_pct = mining_stats(z.detach(), y, MARGIN)  # returns e.g. 33.4 (%)
            train_active_sum += act_pct
            train_batches += 1

            a, p, n = mine_batch(z.detach(), y, MARGIN)
            if len(a) == 0: continue

            loss = triplet_loss(z[a], z[p], z[n])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_triplets += len(a)
            contrib_batches += 1

        sched.step()
        avg_loss = total_loss / max(1, contrib_batches)

        # ---------- VALID ----------
        model.eval()
        with torch.no_grad():
            # (A) collect ALL val embeddings to stabilize stats
            all_z, all_y = [], []
            val_total_loss, val_total_triplets, val_contrib_batches = 0.0, 0, 0

            for x, y, _, _ in val_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                z = model(x)
                all_z.append(z)
                all_y.append(y)

                a, p, n = mine_batch(z, y, MARGIN)
                if len(a) == 0: continue
                loss = triplet_loss(z[a], z[p], z[n])
                
                val_total_loss += loss.item()
                val_total_triplets += len(a)
                val_contrib_batches += 1

            # (B) compute active% ONCE over the whole val set
            if all_z:
                Z = torch.cat(all_z, dim=0)
                Y = torch.cat(all_y, dim=0)
                val_act_pct = mining_stats(Z, Y, MARGIN)  # returns e.g. 33.4 (%)
                val_active_sum += val_act_pct
                val_batches += 1

        val_avg_loss = val_total_loss / max(1, val_contrib_batches)

        # ----- best/early stop -----
        improved = (best_val_loss - val_avg_loss) > ES_MIN_DELTA
        if improved:
            best_val_loss = val_avg_loss
            es_bad_epochs = 0
            torch.save({'category': CATEGORY, 'state_dict': model.state_dict()}, os.path.join(TRIPLETPATH))
        else:
            es_bad_epochs += 1

        # epoch-level active% for logging (still percentages)
        train_active = train_active_sum / max(1, train_batches)
        val_active   = val_active_sum   / max(1, val_batches)
        val_active_frac = val_active / 100.0

        # ----- adaptive margin (convert to FRACTION here) -----
        if ema_val_active is None:
            ema_val_active = val_active_frac
        else:
            ema_val_active = EMA_BETA * ema_val_active + (1 - EMA_BETA) * val_active_frac

        if cooldown == 0:
            if ema_val_active < TARGET_LOW:
                MARGIN = min(M_MAX, MARGIN + STEP)
                cooldown = COOLDOWN_EPOCHS
                triplet_loss.margin = MARGIN
            elif ema_val_active > TARGET_HIGH:
                MARGIN = max(M_MIN, MARGIN - STEP)
                cooldown = COOLDOWN_EPOCHS
                triplet_loss.margin = MARGIN
        else:
            cooldown -= 1

        print(
            f"Epoch[{epoch+1}/{TOTAL_EPOCHS}] "
            f"- train_loss={avg_loss:.4f} - train_triplets={total_triplets} "
            f"- train_active={train_active:.1f}% "
            f"- val_loss={val_avg_loss:.4f} - val_triplets={val_total_triplets} "
            f"- val_active={val_active:.1f}% "
            f"- margin={MARGIN:.4f}"
        )

        # ----- early stopping -----
        if es_bad_epochs >= ES_PATIENCE:
            print(f"Early stopping: no val_loss improvement > {ES_MIN_DELTA} for {ES_PATIENCE} epochs.")
            break


    print("Triplet learning completed.")

    
def load_tl_training_datasets(cfg, paths):
    print("Loading triplet training datasets...")
    BATCH_SIZE = cfg.batch_size
    SEED = 123
    VAL_PERC = 0.2
    
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

from torch.utils.data import Sampler

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

    
if __name__ == "__main__":
    main()
