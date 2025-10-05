from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Hardcoded base path
ARTIFACTS_BASE = Path("/kaggle/working/artifacts")

def make_run_id(category: str) -> str:
    """Generate run_id as YYYY-MM-DD_HH-MM-SS_<category>."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{category.lower()}"

def _ensure_dir(path: Path):
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)

def get_paths(category: str, proc: str):
    """Return namespaces for Student, Triplet, and Pipeline stages (with dirs created)."""
    category = category.lower()
    run_id = make_run_id(category)

    # Build paths
    if proc not in ["student", "triplet", "pipeline"]:
        raise ValueError(f"Invalid proc '{proc}'. Must be one of ['student', 'triplet', 'pipeline'].")
    elif proc == "student":
        root = ARTIFACTS_BASE / "student" / category / run_id
        ns = SimpleNamespace(
            root=root,
            checkpoint=root / "student_best.pth.tar",
            calibration=root / "calibration_stats.npz",
            config=root / "config.yaml",
        )
    elif proc == "triplet":
        root = ARTIFACTS_BASE / "triplet" / category / run_id
        ns = SimpleNamespace(
            root=root,
            checkpoint=root / "triplet_best.pth.tar",
            logs=root / "train_log.json",
            manifest=root / "crop_manifest.jsonl",
            config=root / "config.yaml",
            metrics = root / "metrics.npz",
            gallery = root / "gallery_embeddings.pt",
            tsne = root / "TSNE.png",
        )
    elif proc == "pipeline":
        root = ARTIFACTS_BASE / "pipeline" / category / run_id
        ns = SimpleNamespace(
            root=root,
            predictions=root / "pipeline_predictions.jsonl",
            metrics=root / "pipeline_metrics.json",
            config=root / "config.yaml",
        )

    # Ensure directories exist
    _ensure_dir(root)

    return ns
