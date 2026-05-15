"""Part 3 — Exploration pipeline.

Chains: mask refinement (SAM3/GrabCut) → diffusion inpainting (SD/enhanced fallback).
Builds on Part 2 masks and compares against Part 1 & Part 2 results.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_io import load_video_or_frames, frames_to_video, get_source_info
from utils.mask_utils import save_masks, load_masks
from part3_exploration.sam3_upgrade import get_mask_refiner
from part3_exploration.diffusion_inpaint import (
    get_diffusion_inpainter, DiffusionInpainter, EnhancedInpainter,
    check_diffusion_available,
)


def run_pipeline(
    video_path: str,
    masks_dir: str = "results/part2/masks",
    output_dir: str = "results/part3",
    save_intermediate: bool = True,
    use_diffusion: bool = True,
) -> None:
    """Run the Part 3 exploration pipeline.

    Expects Part 2 masks to already exist (run Part 2 first).
    When *use_diffusion* is True and diffusers is installed, uses
    Stable Diffusion keyframe inpainting; otherwise falls back to
    multi-scale cv2 inpainting with temporal blending.
    """
    os.makedirs(output_dir, exist_ok=True)
    info = get_source_info(video_path)
    fps = info["fps"]

    print("[1/3] Loading frames and masks...")
    frames = load_video_or_frames(video_path)

    if os.path.isdir(masks_dir):
        masks = load_masks(masks_dir)
    else:
        print(f"Masks dir not found: {masks_dir}")
        print("Running Part 2 tracker to generate masks...")
        from part2_sota.sam2_tracker import get_tracker, FallbackTracker
        tracker = get_tracker(use_sam2=True)
        if isinstance(tracker, FallbackTracker):
            masks = tracker.track(frames)
        else:
            raise RuntimeError("Please run Part 2 pipeline first to generate masks")

    print(f"  → {len(frames)} frames, {len(masks)} masks")

    print("[2/3] Refining masks...")
    refiner = get_mask_refiner()
    refined_masks = refiner.refine_masks(frames, masks)
    if save_intermediate:
        save_masks(refined_masks, os.path.join(output_dir, "masks_refined"))

    print("[3/3] Running advanced inpainting...")
    if use_diffusion and check_diffusion_available():
        inpainter = DiffusionInpainter()
    else:
        inpainter = get_diffusion_inpainter(use_diffusion=False)
    inpainted = inpainter.inpaint(frames, refined_masks)

    out_path = os.path.join(output_dir, "inpainted.mp4")
    frames_to_video(inpainted, out_path, fps=fps)
    print(f"Done! Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 3 Exploration Pipeline")
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", default="results/part2/masks", help="Part 2 mask directory")
    parser.add_argument("--output", default="results/part3")
    parser.add_argument("--no-intermediate", action="store_true")
    parser.add_argument("--use-diffusion", action="store_true", default=True,
                        help="Use Stable Diffusion inpainting (default: True)")
    parser.add_argument("--no-diffusion", action="store_true",
                        help="Force fallback inpainter instead of diffusion")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        masks_dir=args.masks,
        output_dir=args.output,
        save_intermediate=not args.no_intermediate,
        use_diffusion=not args.no_diffusion,
    )
