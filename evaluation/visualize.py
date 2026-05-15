"""Visualization utilities: side-by-side comparisons, overlay, and metric plots."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Optional
from pathlib import Path


def create_comparison(
    original: np.ndarray,
    mask: np.ndarray,
    inpainted: np.ndarray,
    title: str = "",
) -> np.ndarray:
    """Create a side-by-side comparison image: original | mask overlay | inpainted."""
    h, w = original.shape[:2]

    overlay = original.copy()
    if mask.max() > 0:
        red_mask = np.zeros_like(original)
        red_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

    comparison = np.hstack([original, overlay, inpainted])

    if title:
        cv2.putText(comparison, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return comparison


def save_comparison_grid(
    originals: List[np.ndarray],
    masks: List[np.ndarray],
    results_dict: Dict[str, List[np.ndarray]],
    output_path: str,
    frame_indices: Optional[List[int]] = None,
    max_frames: int = 6,
) -> None:
    """Save a grid comparing original vs multiple methods.

    Args:
        results_dict: {"Part1": [frames], "Part2": [frames], "Part3": [frames]}
    """
    n_methods = len(results_dict)
    if frame_indices is None:
        total = len(originals)
        step = max(1, total // max_frames)
        frame_indices = list(range(0, total, step))[:max_frames]

    n_frames = len(frame_indices)
    fig, axes = plt.subplots(n_frames, n_methods + 1, figsize=(4 * (n_methods + 1), 3 * n_frames))
    if n_frames == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(frame_indices):
        orig_rgb = cv2.cvtColor(originals[idx], cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(orig_rgb)
        axes[row, 0].set_title(f"Original (#{idx})" if row == 0 else f"#{idx}")
        axes[row, 0].axis("off")

        for col, (name, frames) in enumerate(results_dict.items(), 1):
            if idx < len(frames):
                result_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(result_rgb)
            axes[row, col].set_title(name if row == 0 else "")
            axes[row, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved to: {output_path}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """Bar chart comparing metrics across methods."""
    methods = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    metric_names = [m for m in metric_names if "std" not in m]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for ax, metric in zip(axes, metric_names):
        values = [metrics_dict[m].get(metric, 0) for m in methods]
        std_key = metric.replace("_mean", "_std")
        stds = [metrics_dict[m].get(std_key, 0) for m in methods]

        bars = ax.bar(methods, values, yerr=stds, color=colors, capsize=5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Score")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Metrics comparison saved to: {output_path}")


def create_mask_overlay_video(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    color: tuple = (0, 0, 255),
    alpha: float = 0.4,
) -> None:
    """Create a video with mask overlay for visualization."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import frames_to_video

    overlays = []
    for frame, mask in zip(frames, masks):
        overlay = frame.copy()
        if mask.max() > 0:
            colored = np.zeros_like(frame)
            colored[:] = color
            mask_3ch = np.stack([mask > 128] * 3, axis=-1)
            overlay = np.where(
                mask_3ch,
                (alpha * colored + (1 - alpha) * frame).astype(np.uint8),
                frame,
            )
        overlays.append(overlay)

    frames_to_video(overlays, output_path, fps=fps)
    print(f"Overlay video saved to: {output_path}")
