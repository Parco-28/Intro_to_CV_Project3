"""Part 2 — SOTA pipeline.

Chains: SAM 2 tracking (or fallback) → ProPainter inpainting (or flow-guided fallback).
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_io import load_video_or_frames, frames_to_video, get_source_info
from utils.mask_utils import save_masks
from part2_sota.sam2_tracker import get_tracker, SAM2Tracker, FallbackTracker
from part2_sota.propainter_inpaint import get_inpainter


def run_pipeline(
    video_path: str,
    output_dir: str = "results/part2",
    use_sam2: bool = True,
    use_propainter: bool = True,
    save_intermediate: bool = True,
) -> None:
    """Run the full Part 2 SOTA pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    info = get_source_info(video_path)
    fps = info["fps"]

    print("[1/3] Loading frames...")
    frames = load_video_or_frames(video_path)
    print(f"  -> {len(frames)} frames @ {fps:.1f} FPS")

    # Step 2: Tracking / mask extraction
    print("[2/3] Extracting masks...")
    tracker = get_tracker(use_sam2=use_sam2)

    if isinstance(tracker, FallbackTracker):
        masks = tracker.track(frames)
    else:
        import cv2
        tmp_dir = os.path.join(output_dir, "frames_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])

        from part1_baseline.detect_and_segment import load_model, detect_frame
        model = load_model()

        n = len(frames)
        seed_indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        seed_masks, valid_indices = [], []
        for idx in seed_indices:
            m, _ = detect_frame(model, frames[idx])
            if m.max() > 0:
                seed_masks.append(m)
                valid_indices.append(idx)

        if not valid_indices:
            seed_masks = [np.zeros(frames[0].shape[:2], dtype=np.uint8)]
            valid_indices = [0]

        masks = tracker.track_from_detections(tmp_dir, seed_masks, valid_indices)

    if save_intermediate:
        save_masks(masks, os.path.join(output_dir, "masks"))
    print(f"  → {sum(1 for m in masks if m.max() > 0)} frames with detections")

    # Step 3: Inpainting
    print("[3/3] Inpainting...")
    inpainter = get_inpainter(use_propainter=use_propainter)
    inpainted = inpainter.inpaint(frames, masks, output_dir=output_dir)

    out_path = os.path.join(output_dir, "inpainted.mp4")
    frames_to_video(inpainted, out_path, fps=fps)
    print(f"Done! Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 SOTA Pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/part2")
    parser.add_argument("--no-sam2", action="store_true")
    parser.add_argument("--no-propainter", action="store_true")
    parser.add_argument("--no-intermediate", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        use_sam2=not args.no_sam2,
        use_propainter=not args.no_propainter,
        save_intermediate=not args.no_intermediate,
    )
