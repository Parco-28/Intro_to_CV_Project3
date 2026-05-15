"""Part 1 — Traditional inpainting using OpenCV.

Supports Telea and Navier-Stokes methods with temporal background propagation:
prioritises borrowing clean pixels from neighboring frames via optical flow
warping, then falls back to spatial cv2.inpaint for remaining holes.
"""

import cv2
import numpy as np
from typing import List, Optional

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.mask_utils import dilate_mask


def _warp_frame(src: np.ndarray, src_gray: np.ndarray, dst_gray: np.ndarray) -> np.ndarray:
    """Warp *src* to align with *dst* using dense Farneback optical flow."""
    flow = cv2.calcOpticalFlowFarneback(
        src_gray, dst_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = src_gray.shape
    coords = np.mgrid[0:h, 0:w].astype(np.float32)
    map_y = coords[0] + flow[..., 1]
    map_x = coords[1] + flow[..., 0]
    return cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)


def inpaint_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: int = 5,
    dilate_kernel: int = 15,
) -> np.ndarray:
    """Inpaint a single frame using cv2.inpaint (no temporal info)."""
    if dilate_kernel > 0:
        mask = dilate_mask(mask, kernel_size=dilate_kernel)

    mask_bin = (mask > 128).astype(np.uint8)

    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(frame, mask_bin, radius, flag)


def inpaint_frame_temporal(
    frame: np.ndarray,
    mask: np.ndarray,
    neighbors: List[np.ndarray],
    neighbor_masks: List[np.ndarray],
    frame_gray: np.ndarray,
    neighbor_grays: List[np.ndarray],
    method: str = "telea",
    radius: int = 5,
    dilate_kernel: int = 15,
) -> np.ndarray:
    """Inpaint using temporal background propagation.

    For each masked pixel, try to borrow a clean (unmasked) pixel from the
    nearest neighbor frame after optical-flow alignment.  Only pixels that
    are *not* masked in the neighbor frame are considered clean.
    Remaining holes fall back to spatial cv2.inpaint.
    """
    if dilate_kernel > 0:
        mask = dilate_mask(mask, kernel_size=dilate_kernel)

    if mask.max() == 0:
        return frame

    filled = frame.copy()
    remaining = mask.copy()

    for nb_frame, nb_mask, nb_gray in zip(neighbors, neighbor_masks, neighbor_grays):
        if remaining.max() == 0:
            break

        warped = _warp_frame(nb_frame, nb_gray, frame_gray)
        warped_mask = _warp_frame(
            nb_mask[:, :, np.newaxis] if nb_mask.ndim == 2 else nb_mask,
            nb_gray, frame_gray,
        )
        if warped_mask.ndim == 3:
            warped_mask = warped_mask[:, :, 0]

        clean_in_neighbor = warped_mask < 128
        need_fill = remaining > 128
        usable = clean_in_neighbor & need_fill

        if usable.any():
            usable_3ch = np.stack([usable] * 3, axis=-1)
            filled[usable_3ch] = warped[usable_3ch]
            remaining[usable] = 0

    if remaining.max() > 0:
        mask_bin = (remaining > 128).astype(np.uint8)
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        filled = cv2.inpaint(filled, mask_bin, radius, flag)

    return filled


def inpaint_video(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    method: str = "telea",
    radius: int = 5,
    dilate_kernel: int = 15,
    temporal: bool = True,
    search_range: int = 5,
) -> List[np.ndarray]:
    """Inpaint all frames.

    When *temporal* is True (default), uses neighboring frames to borrow
    clean background pixels before falling back to cv2.inpaint.
    """
    if len(frames) != len(masks):
        raise ValueError("frames and masks must have the same length")

    if not temporal:
        return [inpaint_frame(f, m, method, radius, dilate_kernel)
                for f, m in zip(frames, masks)]

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    results = []

    for i in range(len(frames)):
        if masks[i].max() == 0:
            results.append(frames[i])
            continue

        nb_indices = []
        for offset in range(1, search_range + 1):
            for j in [i - offset, i + offset]:
                if 0 <= j < len(frames):
                    nb_indices.append(j)

        neighbors = [frames[j] for j in nb_indices]
        neighbor_masks = [masks[j] for j in nb_indices]
        neighbor_grays = [grays[j] for j in nb_indices]

        results.append(inpaint_frame_temporal(
            frames[i], masks[i],
            neighbors, neighbor_masks,
            grays[i], neighbor_grays,
            method=method, radius=radius, dilate_kernel=dilate_kernel,
        ))

    return results


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import extract_frames, frames_to_video, get_video_info
    from utils.mask_utils import load_masks

    parser = argparse.ArgumentParser(description="CV2 inpainting")
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True, help="Directory of masks")
    parser.add_argument("--output", default="results/part1/inpainted.mp4")
    parser.add_argument("--method", default="telea", choices=["telea", "ns"])
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--no-temporal", action="store_true", help="Disable temporal propagation")
    parser.add_argument("--search-range", type=int, default=5, help="Neighbor search range")
    args = parser.parse_args()

    frames = extract_frames(args.video)
    masks = load_masks(args.masks)
    info = get_video_info(args.video)
    print(f"Frames: {len(frames)}, Masks: {len(masks)}, FPS: {info['fps']}")

    inpainted = inpaint_video(
        frames, masks, method=args.method, radius=args.radius,
        temporal=not args.no_temporal, search_range=args.search_range,
    )
    frames_to_video(inpainted, args.output, fps=info["fps"])
    print(f"Inpainted video saved to: {args.output}")
