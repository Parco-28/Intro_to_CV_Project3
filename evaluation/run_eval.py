"""Run a complete evaluation across all three pipelines on a sample.

This script:
  1. Loads the original frames and the ProPainter-supplied ground-truth masks.
  2. Compares the masks predicted by Part 1 / Part 2 / Part 3 against the GT.
  3. Reports PSNR / SSIM of each inpainting result against the *original*
     frames, restricted to the masked regions (where the difference matters).
  4. Saves a JSON report and a multi-method comparison grid.

Note: because the bmx-trees / tennis samples bundled with ProPainter do not
provide clean backgrounds, the inpainting PSNR is reported against the
*original* frames inside the masked region only; this measures how much the
restored pixels diverge from the (occluded) original signal. It is the same
protocol used by ProPainter when reporting "in-frame" metrics and is what we
report in the paper.

Usage
-----
    python evaluation/run_eval.py --sample bmx-trees
    python evaluation/run_eval.py --sample tennis --out results/eval/tennis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.mask_utils import load_masks, load_davis_masks
from utils.video_io import extract_frames, load_video_or_frames
from evaluation.metrics import (
    evaluate_inpainting,
    evaluate_masks,
    print_metrics,
)
from evaluation.visualize import save_comparison_grid


def _resolve_inpainted(part_dir: Path) -> Optional[List[np.ndarray]]:
    """Find an inpainted video / frame directory under a part_X/<sample>/ dir."""
    mp4 = part_dir / "inpainted.mp4"
    if mp4.is_file():
        return extract_frames(str(mp4))
    for sub in part_dir.iterdir():
        if sub.is_dir():
            frames_dir = sub / "frames"
            if frames_dir.is_dir():
                return load_video_or_frames(str(frames_dir))
    return None


def _resolve_masks(part_dir: Path) -> Optional[List[np.ndarray]]:
    for name in ("masks_refined", "masks"):
        d = part_dir / name
        if d.is_dir():
            return load_masks(str(d))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="bmx-trees", help="Sample name (bmx-trees, tennis, ...)")
    ap.add_argument("--data-root", default="data/sample")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--gt-mask-dir", default=None,
                    help="Optional GT mask directory (e.g. data/sample/bmx-trees_mask)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-grid", default=None,
                    help="If set, save a method comparison grid PNG here")
    args = ap.parse_args()

    sample = args.sample
    data_root = ROOT / args.data_root
    res_root = ROOT / args.results_root

    frames = load_video_or_frames(str(data_root / sample))
    print(f"[load] {len(frames)} original frames")

    gt_mask_dir = Path(args.gt_mask_dir) if args.gt_mask_dir else data_root / f"{sample}_mask"
    if gt_mask_dir.is_dir():
        try:
            gt_masks = load_davis_masks(str(gt_mask_dir))
        except Exception:
            gt_masks = load_masks(str(gt_mask_dir))
    else:
        gt_masks = None
    if gt_masks is not None:
        print(f"[load] {len(gt_masks)} GT masks from {gt_mask_dir}")

    report: Dict[str, Dict[str, float]] = {}
    inpainted_by_method: Dict[str, List[np.ndarray]] = {}
    mask_by_method: Dict[str, Optional[List[np.ndarray]]] = {}

    for part in ("part1", "part2", "part3"):
        part_dir = res_root / part / sample
        if not part_dir.is_dir():
            print(f"[skip] {part_dir} not found")
            continue

        print(f"\n=== {part} ===")
        masks = _resolve_masks(part_dir)
        inpainted = _resolve_inpainted(part_dir)

        entry: Dict[str, float] = {}

        if masks is not None and gt_masks is not None:
            m_metrics = evaluate_masks(masks[: len(gt_masks)], gt_masks[: len(masks)])
            print_metrics(m_metrics, f"{part} masks vs GT")
            entry.update({f"mask/{k}": v for k, v in m_metrics.items()})

        if inpainted is not None:
            n = min(len(inpainted), len(frames))
            inp_metrics = evaluate_inpainting(inpainted[:n], frames[:n], masks=masks[:n] if masks else None)
            print_metrics(inp_metrics, f"{part} inpainting vs original")
            entry.update({f"inpaint/{k}": v for k, v in inp_metrics.items()})

        report[part] = entry
        if inpainted is not None:
            inpainted_by_method[part] = inpainted
        mask_by_method[part] = masks

    out_path = Path(args.out) if args.out else res_root / "eval" / f"{sample}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report -> {out_path}")

    if args.save_grid and inpainted_by_method:
        ref_mask = mask_by_method.get("part2") or mask_by_method.get("part1") or [np.zeros_like(frames[0][..., 0]) for _ in frames]
        save_comparison_grid(
            frames, ref_mask, inpainted_by_method,
            output_path=args.save_grid, max_frames=4,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
