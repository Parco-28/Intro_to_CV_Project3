"""Sanity check for the project environment.

Run after installing dependencies to verify everything is in place.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, label: str | None = None) -> bool:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK] {label or name:<20s} {ver}")
        return True
    except Exception as e:
        print(f"  [--] {label or name:<20s} {e}")
        return False


def check_file(path: Path, label: str) -> bool:
    if path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {label:<35s} {size_mb:.1f} MB")
        return True
    print(f"  [--] {label:<35s} MISSING ({path})")
    return False


def main() -> int:
    print("=" * 60)
    print(" Environment check")
    print("=" * 60)
    print(f"  Python: {sys.version.split()[0]} ({sys.executable})")

    print("\n[1] Core packages:")
    check("numpy")
    check("cv2", "opencv-python")
    check("scipy")
    check("skimage", "scikit-image")
    check("matplotlib")
    check("tqdm")
    check("PIL", "Pillow")

    print("\n[2] Deep learning stack:")
    has_torch = check("torch")
    if has_torch:
        import torch
        print(f"       CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"       Device: {torch.cuda.get_device_name(0)}")
            print(f"       CUDA version: {torch.version.cuda}")
    check("torchvision")
    check("ultralytics")
    check("sam2")
    check("diffusers")

    print("\n[3] Model weights:")
    check_file(ROOT / "third_party" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt",
               "SAM 2.1 base+ checkpoint")
    pp_weights = ROOT / "third_party" / "ProPainter" / "weights"
    check_file(pp_weights / "ProPainter.pth", "ProPainter weight")
    check_file(pp_weights / "raft-things.pth", "RAFT weight")
    check_file(pp_weights / "recurrent_flow_completion.pth", "Flow-completion weight")

    print("\n[4] Sample data:")
    sample = ROOT / "data" / "sample"
    for name in ["bmx-trees", "bmx-trees_mask", "tennis", "tennis_mask"]:
        d = sample / name
        if d.is_dir():
            n = sum(1 for _ in d.iterdir() if _.suffix.lower() in {".jpg", ".png"})
            print(f"  [OK] data/sample/{name:<20s} {n} frames")
        else:
            print(f"  [--] data/sample/{name:<20s} MISSING")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
