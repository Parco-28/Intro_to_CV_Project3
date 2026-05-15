"""Generate a quick preview PNG showing original / mask / inpainted side by side
for a handful of representative frames.

Usage:
    python scripts/quick_preview.py \
        --frames data/sample/bmx-trees \
        --masks  results/part1/bmx-trees/masks_refined \
        --inpainted results/part1/bmx-trees/inpainted.mp4 \
        --output results/part1/bmx-trees/preview.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.video_io import load_video_or_frames
from utils.mask_utils import load_masks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--inpainted", required=True, help="video or directory")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    frames = load_video_or_frames(args.frames)
    masks = load_masks(args.masks)
    inpainted = load_video_or_frames(args.inpainted)

    n = min(args.n, len(frames), len(masks), len(inpainted))
    step = max(1, len(frames) // n)
    idxs = [i * step for i in range(n)]

    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, i in enumerate(idxs):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        if masks[i].max() > 0:
            mask_3 = np.stack([masks[i] > 128] * 3, axis=-1)
            red = np.zeros_like(rgb)
            red[..., 0] = 255
            overlay = np.where(mask_3, (0.6 * red + 0.4 * rgb).astype(np.uint8), rgb)
        inp = cv2.cvtColor(inpainted[i], cv2.COLOR_BGR2RGB)

        for ax, img, label in zip(axes[row], [rgb, overlay, inp],
                                   ["original", "mask overlay", "inpainted"]):
            ax.imshow(img)
            ax.set_title(f"{label}  (#{i})" if row == 0 else f"#{i}")
            ax.axis("off")

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
