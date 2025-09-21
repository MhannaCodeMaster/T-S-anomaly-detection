import argparse
from datetime import datetime
from pathlib import Path
import yaml

def load_config(config_path):
    """
    Loads configuration from a YAML file .

    Returns:
        dict: Configuration loaded from the YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                raise
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        print(f"Unexpected error reading config file: {e}")
        raise

    return config

def load_args():
    """
    Loads command line arguments.
    Returns: 
        args: parsed command line arguments.
    """
    p = argparse.ArgumentParser(description="Anomaly Detection")
    #----- Required args -----#
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"], default="train", help="Operation mode: train or test")
    p.add_argument("--config", type=str, required=True, default="conf/config.yaml", help="Path to config file")
    p.add_argument("--dataset.mvtec", required=True, type=str, help="Path to mvtec dataset root directory")
    p.add_argument("--dataset.crops_root", required=True, type=str, help="Path to the crops directory")
    p.add_argument("--dataset.ok_manifest", required=True, type=str, help="Path to ok crops manifest file")
    p.add_argument("--dataset.notok_manifest", required=True, type=str, help="Path to not ok crops manifest file")
    
    #----- Optional args -----#
    p.add_argument("--dataset.category", required=False, type=str, help="Dataset category (e.g., cable, hazelnut)")
    p.add_argument("--student.train", type=bool, required=False, help="Train stdudent model")
    p.add_argument("--student.epochs", type=int, required=False, help="Number of student training epochs")
    p.add_argument("--student.batch_size", type=int, required=False, help="Batch size for student training")
    p.add_argument("--triplet.train", type=bool, required=False, help="Train triplet model")
    p.add_argument("--triplet.epochs", type=int, required=False, help="Number of triplet training epochs")
    p.add_argument("--triplet.batch_size", type=int, required=False, help="Batch size for triplet training")
    p.add_argument("--heatmap_threshold.method", type=int, required=False,choices=["percentile", "otsu"], help="Heatmap thresholding method")
    p.add_argument("--heatmap_threshold.value", type=float, required=False, help="Heatmap threshold value if method percentile")
    p.add_argument("--models.st_path", type=str, required=False, help="Path to student model checkpoint")
    p.add_argument("--models.tl_path", type=str, required=False, help="Path to triplet learning model checkpoint")
    p.add_argument("--models.calibration", type=str, required=False, help="Path to the calibration variables")
    p.add_argument("--box.expand", type=float, required=False,  default=0.5, help="Box expansion ratio")
    p.add_argument("--box.min_area", type=int, required=False, default=600, help="Minimum area for box filtering")
    p.add_argument("--nms.mode", type=str, required=False, choices=["mean", "max"], default="mean", help="NMS scoring mode")
    p.add_argument("--nms.score_thr", type=float, required=False, default=0.1, help="NMS score threshold")
    p.add_argument("--nms.thr", type=float, required=False, default=0.4, help="NMS IoU threshold")

    args = p.parse_args()
    return args

def set_deep(cfg, dotted_key, value):
    """Set cfg['a']['b']['c'] = value if dotted_key='a.b.c'."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}   # create if missing
        d = d[k]
    d[keys[-1]] = value

def apply_overrides(cfg, args):
    for key, val in vars(args).items():
        if val is None or key == "config":
            continue
        set_deep(cfg, key, val)
    return cfg

def get_config():
    args = load_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    return cfg

def resolve_paths(cfg):
    """Create timestamped run directories under paths.root/mode/category/<stamp>."""
    stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = Path("outputs")
    base = root / cfg['mode'] / cfg['dataset']['category'] / stamp
    out = {
        'base': base,
        'images':  base / 'images',
        'crops':   base / 'crops',
        'student': base / 'student',
        'triplet': base / 'triplet',
        'calibration': base / 'calibration'
    }
    # Make dirs
    for k in ('images','crops','student','triplet','calibration'):
        out[k].mkdir(parents=True, exist_ok=True)
    return out

def set_deep(cfg, dotted_key, value):
    """Set cfg['a']['b']['c'] = value if dotted_key='a.b.c'."""
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d:
            d[k] = {}   # create if missing
        d = d[k]
    d[keys[-1]] = value

def apply_overrides(cfg, args):
    for key, val in vars(args).items():
        if val is None or key == "config":
            continue
        set_deep(cfg, key, val)
    return cfg
