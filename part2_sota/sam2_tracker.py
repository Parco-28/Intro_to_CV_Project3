"""Part 2 — SAM 2 based video object tracking and mask extraction.

Uses SAM 2 (Segment Anything Model 2) for temporally consistent
mask generation across video frames.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple


def check_sam2_available() -> bool:
    """Check if SAM 2 is installed."""
    try:
        from sam2.build_sam import build_sam2_video_predictor
        return True
    except ImportError:
        return False


DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_CKPT_NAME = "sam2.1_hiera_base_plus.pt"


def _resolve_sam2_checkpoint(ckpt: str) -> str:
    """Try a few likely locations to find a SAM2 checkpoint file."""
    p = Path(ckpt)
    if p.is_file():
        return str(p)
    candidates = [
        Path(__file__).resolve().parent.parent / "third_party" / "sam2" / "checkpoints" / p.name,
        Path(__file__).resolve().parent.parent / "weights" / p.name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(ckpt)


class SAM2Tracker:
    """Video object tracker using SAM 2."""

    def __init__(self, model_cfg: str = DEFAULT_SAM2_CFG, checkpoint: str = DEFAULT_SAM2_CKPT_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cfg = model_cfg
        self.checkpoint = _resolve_sam2_checkpoint(checkpoint)
        self.predictor = None

    def load_model(self):
        """Load SAM 2 video predictor."""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, self.checkpoint, device=self.device
            )
            print(f"SAM 2 loaded on {self.device}: {self.checkpoint}")
        except ImportError:
            raise RuntimeError(
                "SAM 2 not installed. Install with:\n"
                "  pip install -e third_party/sam2"
            )

    def track_from_detections(
        self,
        frames_dir: str,
        initial_masks: List[np.ndarray],
        frame_indices: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Track objects across video using initial detection masks as prompts.

        Args:
            frames_dir: Directory containing video frames as images.
            initial_masks: Initial masks to use as prompts (from YOLOv8 or manual).
            frame_indices: Which frames the initial_masks correspond to.

        Returns:
            List of binary masks for each frame.
        """
        if self.predictor is None:
            self.load_model()

        inference_state = self.predictor.init_state(video_path=frames_dir)

        if frame_indices is None:
            frame_indices = [0]

        for idx, mask in zip(frame_indices, initial_masks):
            if mask.max() == 0:
                continue
            _, _, _ = self.predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=idx,
                obj_id=1,
                mask=mask.astype(bool),
            )

        video_segments = {}
        for frame_idx, obj_ids, masks in self.predictor.propagate_in_video(inference_state):
            if len(masks) > 0:
                combined = np.zeros_like(masks[0][0].cpu().numpy(), dtype=np.uint8)
                for m in masks:
                    binary = (m[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    combined = cv2.bitwise_or(combined, binary)
                video_segments[frame_idx] = combined
            else:
                video_segments[frame_idx] = None

        frame_files = sorted([p for p in Path(frames_dir).iterdir()
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        total_frames = len(frame_files)
        h, w = cv2.imread(str(frame_files[0])).shape[:2]
        result_masks = []
        for i in range(total_frames):
            if i in video_segments and video_segments[i] is not None:
                m = video_segments[i]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                result_masks.append(m)
            else:
                result_masks.append(np.zeros((h, w), dtype=np.uint8))

        return result_masks


class FallbackTracker:
    """Fallback tracker using YOLOv8-Seg per-frame detection when SAM 2 is unavailable.

    Applies temporal smoothing for more consistent masks.
    """

    def __init__(self, model_name: str = "yolov8m-seg.pt"):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from part1_baseline.detect_and_segment import load_model
        self.model = load_model(model_name)

    def track(self, frames: List[np.ndarray], conf: float = 0.35) -> List[np.ndarray]:
        """Per-frame detection with temporal smoothing."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from part1_baseline.detect_and_segment import detect_video

        raw_masks = detect_video(self.model, frames, conf_threshold=conf)
        return self._temporal_smooth(raw_masks, window=3)

    @staticmethod
    def _temporal_smooth(masks: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
        """Simple temporal smoothing: majority vote over a sliding window."""
        smoothed = []
        half = window // 2
        for i in range(len(masks)):
            start = max(0, i - half)
            end = min(len(masks), i + half + 1)
            stack = np.stack(masks[start:end], axis=0).astype(np.float32) / 255.0
            avg = np.mean(stack, axis=0)
            smoothed.append(((avg > 0.5) * 255).astype(np.uint8))
        return smoothed


def get_tracker(use_sam2: bool = True, **kwargs):
    """Factory: return SAM2Tracker if available, else FallbackTracker."""
    if use_sam2 and check_sam2_available():
        return SAM2Tracker(**kwargs)
    print("SAM 2 not available, using fallback tracker (YOLOv8 + temporal smoothing)")
    return FallbackTracker()


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import extract_frames
    from utils.mask_utils import save_masks

    parser = argparse.ArgumentParser(description="SAM 2 Video Tracker")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="results/part2/masks")
    parser.add_argument("--no-sam2", action="store_true", help="Force fallback tracker")
    args = parser.parse_args()

    frames = extract_frames(args.video)
    print(f"Loaded {len(frames)} frames")

    tracker = get_tracker(use_sam2=not args.no_sam2)

    if isinstance(tracker, FallbackTracker):
        masks = tracker.track(frames)
    else:
        import tempfile, os
        tmp_dir = tempfile.mkdtemp()
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        from part1_baseline.detect_and_segment import load_model, detect_frame
        model = load_model()
        init_mask, _ = detect_frame(model, frames[0])
        masks = tracker.track_from_detections(tmp_dir, [init_mask], [0])

    save_masks(masks, args.output)
    print(f"Masks saved to: {args.output}")
