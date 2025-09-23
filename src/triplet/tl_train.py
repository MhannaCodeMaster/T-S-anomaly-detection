import os
from types import SimpleNamespace

import torch
from torchvision import transforms

from src.triplet.triplet import TripletEmbedder
from src.data.data_utils import *

from conf.config import *
from src.utils.utils import *
from src.paths import get_paths

def main():
    print("Triplet learning started...")
    args = load_args()
    cfg = load_config()
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
    CATEGORY = cfg.category
    TOTAL_EPOCHS = int(cfg.epochs)
    MARGIN = float(cfg.margin)
    LR = float(cfg.lr)
    WGT_DECAY = float(cfg.weight_decay)
    TRIPLETPATH = paths.checkpoint
    
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

        # average over number of batches that produced triplets
        val_avg_loss = val_total_loss / max(1, val_contrib_batches)
        # ---- Validation END ----
        
        # ---- Save best model ----
        if val_total_triplets > 0 and val_avg_loss < min_err:
            min_err = val_avg_loss
            best_model = {
                'category': CATEGORY,
                "state_dict": model.state_dict()
            }
            torch.save(best_model, os.path.join(TRIPLETPATH))
            
        print(f"Epoch[{epoch+1}/{TOTAL_EPOCHS}] - train_loss={avg_loss:.4f} - train_triplets={total_triplets} - val_loss={val_avg_loss:.4f} - val_triplets={val_total_triplets}", end='\r')

    print("Triplet learning completed.")
    
def load_tl_training_datasets(cfg, paths):
    print("Loading triplet training datasets...")
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
    train_ok_ds    = PatchDataset(train_ok_df, train_tf, "ok", crops_root=cfg.dataset)
    train_notok_ds = PatchDataset(train_notok_df, train_tf, "not_ok", crops_root=cfg.dataset)
    val_ok_ds      = PatchDataset(val_ok_df, val_tf, "ok", crops_root=cfg.dataset)
    val_notok_ds   = PatchDataset(val_notok_df, val_tf, "not_ok", crops_root=cfg.dataset)
    
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

    args = p.parse_args()
    return args

def load_config():
    CONF_PATH = "../conf/triplet.yaml"
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
