"""Part 1 — Object detection & instance segmentation using YOLOv8-Seg.

Produces per-frame binary masks for detected dynamic objects.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ultralytics import YOLO

# COCO classes that are typically considered "dynamic foreground" - i.e. objects
# that move with a person and should be removed together with them. Tennis
# rackets and sports balls in particular are visible in many DAVIS clips and
# must be included, otherwise YOLOv8-Seg leaves obvious ghosting artefacts
# behind the inpainter.
DYNAMIC_CLASSES = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    24,  # backpack (often carried by walking pedestrians)
    25,  # umbrella
    26,  # handbag
    32,  # sports ball
    34,  # baseball bat
    35,  # baseball glove
    37,  # skateboard
    38,  # tennis racket
    39,  # surfboard
}


def load_model(model_name: str = "yolov8m-seg.pt") -> YOLO:
    """Load a YOLOv8-Seg model (downloads weights automatically)."""
    return YOLO(model_name)


def detect_frame(
    model: YOLO,
    frame: np.ndarray,
    conf_threshold: float = 0.35,
    dynamic_only: bool = True,
) -> Tuple[np.ndarray, list]:
    """Run detection on a single frame.

    Returns:
        mask: Combined binary mask (H, W) uint8 with 0/255 values.
        detections: List of dicts with keys cls, conf, bbox, mask.
    """
    h, w = frame.shape[:2]
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    detections = []

    if results.masks is None:
        return combined_mask, detections

    for i, (box, seg_mask) in enumerate(zip(results.boxes, results.masks)):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if dynamic_only and cls_id not in DYNAMIC_CLASSES:
            continue

        mask_resized = cv2.resize(
            seg_mask.data[0].cpu().numpy().astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_binary = (mask_resized > 0).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask_binary)

        detections.append({
            "cls": cls_id,
            "conf": conf,
            "bbox": box.xyxy[0].cpu().numpy().tolist(),
            "mask": mask_binary,
        })

    return combined_mask, detections


def detect_video(
    model: YOLO,
    frames: List[np.ndarray],
    conf_threshold: float = 0.35,
    dynamic_only: bool = True,
) -> List[np.ndarray]:
    """Run detection on all frames, return list of combined binary masks."""
    masks = []
    for frame in frames:
        mask, _ = detect_frame(model, frame, conf_threshold, dynamic_only)
        masks.append(mask)
    return masks


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import extract_frames
    from utils.mask_utils import save_masks

    parser = argparse.ArgumentParser(description="YOLOv8-Seg detection")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="results/part1/masks", help="Mask output dir")
    parser.add_argument("--model", default="yolov8m-seg.pt", help="Model name")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Extracting frames from: {args.video}")
    frames = extract_frames(args.video)
    print(f"Total frames: {len(frames)}")

    print("Running detection...")
    masks = detect_video(model, frames, conf_threshold=args.conf)

    save_masks(masks, args.output)
    print(f"Masks saved to: {args.output}")
