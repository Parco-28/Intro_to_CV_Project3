"""Part 1 — Complete baseline pipeline.

Chains: YOLOv8-Seg detection → optical flow filtering → cv2 inpainting.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm
from utils.video_io import load_video_or_frames, frames_to_video, get_source_info
from utils.mask_utils import save_masks
from part1_baseline.detect_and_segment import load_model, detect_video
from part1_baseline.optical_flow import filter_masks_by_flow
from part1_baseline.inpaint_cv2 import inpaint_video


def run_pipeline(
    video_path: str,
    output_dir: str = "results/part1",
    model_name: str = "yolov8m-seg.pt",
    conf: float = 0.35,
    motion_threshold: float = 2.0,
    inpaint_method: str = "telea",
    inpaint_radius: int = 5,
    save_intermediate: bool = True,
) -> None:
    """Run the full Part 1 baseline pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    info = get_source_info(video_path)
    fps = info["fps"]

    print("[1/4] Loading frames...")
    frames = load_video_or_frames(video_path)
    print(f"  -> {len(frames)} frames @ {fps:.1f} FPS")

    # Step 2: YOLOv8-Seg detection
    print("[2/4] Running YOLOv8-Seg detection...")
    model = load_model(model_name)
    raw_masks = detect_video(model, frames, conf_threshold=conf)
    if save_intermediate:
        save_masks(raw_masks, os.path.join(output_dir, "masks_raw"))

    # Step 3: Optical flow filtering
    print("[3/4] Filtering masks with optical flow...")
    refined_masks = filter_masks_by_flow(frames, raw_masks, motion_threshold=motion_threshold)
    if save_intermediate:
        save_masks(refined_masks, os.path.join(output_dir, "masks_refined"))

    # Step 4: Inpainting
    print("[4/4] Inpainting...")
    inpainted = inpaint_video(frames, refined_masks, method=inpaint_method, radius=inpaint_radius)

    out_path = os.path.join(output_dir, "inpainted.mp4")
    frames_to_video(inpainted, out_path, fps=fps)
    print(f"Done! Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 1 Baseline Pipeline")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="results/part1", help="Output directory")
    parser.add_argument("--model", default="yolov8m-seg.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--motion-threshold", type=float, default=2.0)
    parser.add_argument("--inpaint-method", default="telea", choices=["telea", "ns"])
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--no-intermediate", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        model_name=args.model,
        conf=args.conf,
        motion_threshold=args.motion_threshold,
        inpaint_method=args.inpaint_method,
        inpaint_radius=args.inpaint_radius,
        save_intermediate=not args.no_intermediate,
    )
