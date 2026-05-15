"""Generate publication-style pipeline flowcharts (PNG) for the report.

Run from repo root:
    python scripts/generate_flowcharts.py

Outputs go to report/figures/ (300 DPI, white background).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "report" / "figures"


def _box(ax, xy, w, h, text, fontsize=9, fc="#E8F4FC", ec="#1a5276"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc, mutation_aspect=0.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.2, color="#2c3e50", shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def _setup_ax(title: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#1a252f")
    return fig, ax


def fig_part1():
    fig, ax = _setup_ax("Part 1 — Baseline (hand-crafted)")
    _box(ax, (0.5, 7.2), 2.2, 0.9, "Video / frames", fc="#d5e8d4")
    _box(ax, (3.2, 7.2), 2.6, 0.9, "YOLOv8-Seg\n(dynamic classes)")
    _box(ax, (6.5, 7.2), 2.8, 0.9, "Binary mask\n(per frame)")
    _arrow(ax, 2.7, 7.65, 3.2, 7.65)
    _arrow(ax, 5.8, 7.65, 6.5, 7.65)

    _box(ax, (0.8, 5.0), 3.2, 1.0, "Lucas–Kanade\nsparse optical flow\n+ occlusion-aware filter")
    _arrow(ax, 7.9, 7.2, 2.4, 6.0)

    _box(ax, (4.8, 5.0), 3.8, 1.0, "Mask dilation\n(motion blur)")
    _arrow(ax, 4.0, 5.5, 4.8, 5.5)

    _box(ax, (2.0, 2.5), 5.5, 1.1, "Temporal bg propagation\n(Farneback warp neighbors)\n+ cv2.inpaint fallback")
    _arrow(ax, 4.0, 5.0, 4.75, 3.6)

    _box(ax, (3.0, 0.6), 3.8, 0.85, "Inpainted video (MP4)", fc="#f9e79f")
    _arrow(ax, 4.75, 2.5, 4.75, 1.45)

    fig.tight_layout()
    return fig


def fig_part2():
    fig, ax = _setup_ax("Part 2 — SOTA (SAM 2 + ProPainter)")
    _box(ax, (0.4, 7.2), 2.0, 0.9, "Video / frames", fc="#d5e8d4")
    _box(ax, (2.8, 7.2), 2.8, 0.9, "YOLO seeds\n(multi keyframes)")
    _box(ax, (6.0, 7.2), 3.5, 0.9, "SAM 2 video predictor\n(temporal masks)")
    _arrow(ax, 2.4, 7.65, 2.8, 7.65)
    _arrow(ax, 5.6, 7.65, 6.0, 7.65)

    _box(ax, (1.0, 5.0), 3.5, 1.0, "ProPainter\n(dual-domain propagation\n+ sparse Transformer)")
    _arrow(ax, 7.75, 7.2, 2.75, 6.0)

    _box(ax, (5.2, 5.0), 3.5, 1.0, "Fallback:\nflow-guided inpaint\n(if ProPainter off)")
    _arrow(ax, 4.5, 5.5, 5.2, 5.5)

    _box(ax, (3.0, 2.6), 3.8, 0.95, "Inpainted video", fc="#f9e79f")
    _arrow(ax, 2.75, 5.0, 4.4, 3.55)
    _arrow(ax, 6.95, 5.0, 5.4, 3.55)

    _box(ax, (0.5, 2.6), 2.0, 0.85, "Saved masks\n→ Part 3", fc="#fadbd8")
    _arrow(ax, 7.75, 7.2, 1.5, 3.45)

    fig.tight_layout()
    return fig


def fig_part3():
    fig, ax = _setup_ax("Part 3 — Exploration (refine + generative)")
    _box(ax, (0.4, 7.2), 2.0, 0.9, "Frames", fc="#d5e8d4")
    _box(ax, (2.8, 7.2), 2.8, 0.9, "Part 2 masks\n(or tracker fallback)")
    _arrow(ax, 2.4, 7.65, 2.8, 7.65)

    _box(ax, (6.0, 7.0), 3.6, 1.2, "Mask refinement\n(morphology + GrabCut)\n[SAM 3 path if available]")
    _arrow(ax, 5.6, 7.65, 6.0, 7.65)

    _box(ax, (1.2, 4.6), 3.8, 1.15, "Stable Diffusion inpainting\n(keyframes + optical-flow\npropagation to other frames)")
    _arrow(ax, 7.8, 7.0, 3.1, 5.75)

    _box(ax, (5.5, 4.6), 3.2, 1.15, "Fallback:\nEnhanced multi-scale\ncv2.inpaint + temporal blend")
    _arrow(ax, 5.0, 5.15, 5.5, 5.15)

    _box(ax, (3.0, 2.2), 3.8, 0.9, "Inpainted video", fc="#f9e79f")
    _arrow(ax, 3.1, 4.6, 4.4, 3.1)
    _arrow(ax, 7.1, 4.6, 5.4, 3.1)

    fig.tight_layout()
    return fig


def fig_overview():
    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=150)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("End-to-end overview (three tiers)", fontsize=13, fontweight="bold", pad=8)

    w, h = 3.2, 1.35
    y = 1.9
    _box(ax, (0.3, y), w, h, "Part 1\nDetection + LK flow\n+ temporal cv2.inpaint", fc="#eaf2f8", ec="#2874a6")
    _box(ax, (4.4, y), w, h, "Part 2\nSAM 2 tracking\n+ ProPainter", fc="#eaf2f8", ec="#2874a6")
    _box(ax, (8.5, y), w, h, "Part 3\nMask refine\n+ SD / fallback", fc="#eaf2f8", ec="#2874a6")

    _box(ax, (1.2, 0.35), 9.6, 0.95, "Inputs: bmx-trees, tennis, DAVIS, wild video  •  Metrics: mask IoU/recall, PSNR/SSIM, qualitative grids", fc="#f4f6f6", ec="#7f8c8d")
    _arrow(ax, 3.5, y + h / 2, 4.4, y + h / 2)
    _arrow(ax, 7.6, y + h / 2, 8.5, y + h / 2)

    fig.tight_layout()
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("pipeline_overview", fig_overview),
        ("pipeline_part1", fig_part1),
        ("pipeline_part2", fig_part2),
        ("pipeline_part3", fig_part3),
    ]:
        fig = fn()
        path = OUT_DIR / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
