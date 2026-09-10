"""
fetch_models.py — put the document-layout weights in place.

The two .pt files are ~90 MB and byte-identical to the ones already versioned in
the Zynaptrix backend, so they are gitignored here rather than committed twice.
This script restores them into phoenix/backend/models/.

Usage (from phoenix/backend):
    python scripts/fetch_models.py
"""
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND / "models"

# Sibling checkout of the main product, which tracks these weights.
SIBLING = BACKEND.parent.parent / "backend" / "models"

WEIGHTS = {
    "yolov8_doclaynet.pt": (
        "Document layout detection (YOLOv8 fine-tuned on DocLayNet). Custom "
        "weights — copy from the Zynaptrix backend or your model store."
    ),
    "mobile_sam.pt": (
        "Mobile SAM, used to split composite figures. This is a standard "
        "Ultralytics asset and will be downloaded automatically on first use "
        "if it is missing."
    ),
}


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for name, description in WEIGHTS.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"  ✓ {name} already present ({target.stat().st_size // 1024 // 1024} MB)")
            continue

        source = SIBLING / name
        if source.exists():
            print(f"  → copying {name} from {source} ...")
            shutil.copy2(source, target)
            print(f"  ✓ {name} ({target.stat().st_size // 1024 // 1024} MB)")
        else:
            missing.append((name, description))

    if missing:
        print("\nCould not locate these locally:")
        for name, description in missing:
            print(f"  ✗ {name}\n      {description}")
        print(f"\nPlace them in: {MODELS_DIR}")
        # Mobile SAM self-heals, so only the custom weights are fatal.
        return 1 if any(n == "yolov8_doclaynet.pt" for n, _ in missing) else 0

    print("\nAll layout models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
