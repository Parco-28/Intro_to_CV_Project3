"""Part 1 — Sparse optical flow (Lucas-Kanade) for dynamic object filtering.

Uses feature tracking to distinguish moving objects from static background,
then refines detection masks by removing stationary detections.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)


def compute_flow_magnitude(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute LK optical flow and return magnitudes for tracked points.

    Returns:
        good_old: Successfully tracked points in prev frame (N, 2).
        good_new: Corresponding points in curr frame (N, 2).
        magnitudes: Flow magnitude for each point (N,).
    """
    if points is None or len(points) == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty(0, dtype=np.float32)

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, points, None, **LK_PARAMS
    )
    status = status.flatten().astype(bool)
    good_old = points[status].reshape(-1, 2)
    good_new = next_pts[status].reshape(-1, 2)
    magnitudes = np.linalg.norm(good_new - good_old, axis=1)
    return good_old, good_new, magnitudes


def classify_motion(
    mask: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    motion_threshold: float = 2.0,
    min_moving_ratio: float = 0.3,
    prev_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Refine a detection mask by checking if detected regions are actually moving.

    For each connected component in the mask, sample feature points and
    check their optical flow.  Components with insufficient motion are
    removed **unless** they overlap with a region that was moving in the
    previous frame (occlusion tolerance).
    """
    if mask.max() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    refined = np.zeros_like(mask)
    img_area = mask.shape[0] * mask.shape[1]

    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8) * 255
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < 100:
            continue

        # Small components from a known detector are likely real — keep them
        if area < img_area * 0.005:
            refined = cv2.bitwise_or(refined, component_mask)
            continue

        points = cv2.goodFeaturesToTrack(prev_gray, mask=component_mask, **FEATURE_PARAMS)

        # Few/no trackable points → likely occluded or textureless.
        # If this region overlapped a moving region last frame, keep it.
        if points is None or len(points) < 3:
            if prev_mask is not None and _masks_overlap(component_mask, prev_mask):
                refined = cv2.bitwise_or(refined, component_mask)
            else:
                refined = cv2.bitwise_or(refined, component_mask)
            continue

        _, _, magnitudes = compute_flow_magnitude(prev_gray, curr_gray, points)
        tracked_count = len(magnitudes)

        if tracked_count == 0:
            # All points lost — occlusion likely; keep if previously moving
            if prev_mask is not None and _masks_overlap(component_mask, prev_mask):
                refined = cv2.bitwise_or(refined, component_mask)
            continue

        moving_ratio = np.mean(magnitudes > motion_threshold)

        if moving_ratio >= min_moving_ratio:
            refined = cv2.bitwise_or(refined, component_mask)
        elif prev_mask is not None and _masks_overlap(component_mask, prev_mask):
            # Below motion threshold but was moving last frame → keep
            # (handles brief pauses, partial occlusion)
            refined = cv2.bitwise_or(refined, component_mask)

    return refined


def _masks_overlap(a: np.ndarray, b: np.ndarray, min_iou: float = 0.1) -> bool:
    """Check if two binary masks overlap enough."""
    a_bin = a > 128
    b_bin = b > 128
    intersection = np.logical_and(a_bin, b_bin).sum()
    a_area = a_bin.sum()
    if a_area == 0:
        return False
    return (intersection / a_area) >= min_iou


def filter_masks_by_flow(
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    motion_threshold: float = 2.0,
    min_moving_ratio: float = 0.3,
) -> List[np.ndarray]:
    """Filter a sequence of detection masks using optical flow.

    For the first frame, the mask is kept as-is (no previous frame to compare).
    Each frame receives the previous refined mask for occlusion tolerance.
    """
    if len(frames) != len(masks):
        raise ValueError("frames and masks must have the same length")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    refined = [masks[0].copy()]

    for i in range(1, len(frames)):
        r = classify_motion(
            masks[i], grays[i - 1], grays[i],
            motion_threshold=motion_threshold,
            min_moving_ratio=min_moving_ratio,
            prev_mask=refined[i - 1],
        )
        refined.append(r)

    return refined


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import extract_frames
    from utils.mask_utils import load_masks, save_masks

    parser = argparse.ArgumentParser(description="Optical flow mask filtering")
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True, help="Directory of detection masks")
    parser.add_argument("--output", default="results/part1/masks_refined")
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args()

    frames = extract_frames(args.video)
    masks = load_masks(args.masks)
    print(f"Loaded {len(frames)} frames, {len(masks)} masks")

    refined = filter_masks_by_flow(frames, masks, motion_threshold=args.threshold)
    save_masks(refined, args.output)
    print(f"Refined masks saved to: {args.output}")
