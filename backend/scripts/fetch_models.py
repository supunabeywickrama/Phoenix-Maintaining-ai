"""
fetch_models.py — put the document-layout weights in place.

The two .pt files total ~90 MB and are gitignored, so a fresh clone does not
carry them. This script restores them into backend/models/ from a local source.

Usage (from the backend directory):
    python scripts/fetch_models.py

    # or point it at any directory holding the weights
    set PHOENIX_MODEL_SOURCE=D:\\shared\\models
    python scripts/fetch_models.py
"""
import os
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND / "models"

# Local sources, tried in order: an explicit directory from the environment,
# then a sibling checkout of the Zynaptrix copilot, which versions these weights.
CANDIDATE_DIRS = [
    Path(os.environ["PHOENIX_MODEL_SOURCE"]) if os.environ.get("PHOENIX_MODEL_SOURCE") else None,
    BACKEND.parent.parent / "backend" / "models",
]

WEIGHTS = {
    "yolov8_doclaynet.pt": [
        "Document layout detection — YOLOv8 fine-tuned on DocLayNet (~51 MB).",
        "Custom weights, so there is no automatic download. Copy them from an",
        "existing deployment, or set PHOENIX_MODEL_SOURCE to a directory holding them.",
    ],
    "mobile_sam.pt": [
        "Mobile SAM, used to split composite figures (~39 MB).",
        "A standard Ultralytics asset — fetched automatically on first use if absent,",
        "so this one is safe to leave missing.",
    ],
}

# Only the custom weights are fatal; Mobile SAM self-heals at runtime.
REQUIRED = {"yolov8_doclaynet.pt"}


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for name, description in WEIGHTS.items():
        target = MODELS_DIR / name
        if target.exists():
            size = target.stat().st_size // 1024 // 1024
            print(f"  [ok] {name} already present ({size} MB)")
            continue

        source = next(
            (d / name for d in CANDIDATE_DIRS if d and (d / name).exists()), None
        )
        if source:
            print(f"  ->   copying {name} from {source} ...")
            shutil.copy2(source, target)
            size = target.stat().st_size // 1024 // 1024
            print(f"  [ok] {name} ({size} MB)")
        else:
            missing.append((name, description))

    if missing:
        print("\nCould not locate these locally:")
        for name, description in missing:
            print(f"\n  [--] {name}")
            for line in description:
                print(f"       {line}")
        print(f"\nPlace them in: {MODELS_DIR}")
        if any(name in REQUIRED for name, _ in missing):
            print("\nIngestion will fall back to plain text extraction without them,")
            print("losing figure detection and diagram search.")
            return 1
        return 0

    print("\nAll layout models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
