from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

# Base path
ARTIFACTS_BASE = Path("/kaggle/working/artifacts")

# File names
BEST_STUDENT = "student_best.pth.tar"
CALIBRATION_STATS = "calib_stats.npz"
BEST_TRIPLET = "triplet_best.pth.tar"
CONFIG = "config.yaml"
TL_METRICS = "tl_metrics.npz"
GALLERY_EMBS = "gal_embs.pt"



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
            checkpoint=root / BEST_STUDENT,
            calibration=root / CALIBRATION_STATS,
            config=root / CONFIG,
        )
    elif proc == "triplet":
        root = ARTIFACTS_BASE / "triplet" / category / run_id
        ns = SimpleNamespace(
            root=root,
            checkpoint=root / BEST_TRIPLET,
            config=root / CONFIG,
            metrics = root / TL_METRICS,
            gallery = root / GALLERY_EMBS,
            tsne = root / "tsne.png",
        )
    elif proc == "pipeline":
        root = ARTIFACTS_BASE / "pipeline" / category / run_id
        ns = SimpleNamespace(
            root=root,
            config=root / CONFIG,
        )

    # Ensure directories exist
    _ensure_dir(root)

    return ns
