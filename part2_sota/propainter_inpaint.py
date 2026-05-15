"""Part 2 — ProPainter video inpainting.

Uses ProPainter for temporally coherent video inpainting.
Falls back to flow-guided frame-by-frame inpainting if ProPainter is unavailable.
"""

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from typing import List, Optional


def check_propainter_available() -> bool:
    """Check if ProPainter is installed/accessible (script + weights)."""
    try:
        propainter_path = Path(__file__).resolve().parent.parent / "third_party" / "ProPainter"
        if not (propainter_path / "inference_propainter.py").exists():
            return False
        weights_dir = propainter_path / "weights"
        required = ["ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth"]
        return all((weights_dir / w).exists() for w in required)
    except Exception:
        return False


class ProPainterInpainter:
    """Video inpainting using ProPainter."""

    def __init__(self, propainter_dir: Optional[str] = None):
        if propainter_dir is None:
            self.propainter_dir = Path(__file__).resolve().parent.parent / "third_party" / "ProPainter"
        else:
            self.propainter_dir = Path(propainter_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def inpaint(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        output_dir: str = "results/part2",
    ) -> List[np.ndarray]:
        """Run ProPainter inference via subprocess (official interface).

        Saves frames and masks to temp dirs, calls ProPainter from its own
        directory (its model loads use relative ``weights/`` paths), reads results.
        """
        import sys
        import tempfile
        import subprocess

        tmp_frames = tempfile.mkdtemp(prefix="pp_frames_")
        tmp_masks = tempfile.mkdtemp(prefix="pp_masks_")
        os.makedirs(output_dir, exist_ok=True)
        output_dir_abs = str(Path(output_dir).resolve())

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            cv2.imwrite(os.path.join(tmp_frames, f"{i:05d}.png"), frame)
            cv2.imwrite(os.path.join(tmp_masks, f"{i:05d}.png"), mask)

        cmd = [
            sys.executable,
            str(self.propainter_dir / "inference_propainter.py"),
            "--video", tmp_frames,
            "--mask", tmp_masks,
            "--output", output_dir_abs,
            "--fp16",
        ]

        print(f"Running ProPainter: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(self.propainter_dir), text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ProPainter inference failed (exit {result.returncode})")

        # ProPainter writes its mp4 to ``<output>/<basename(--video)>/inpaint_out.mp4``.
        # We use the newest matching directory in case multiple runs accumulate.
        out_root = Path(output_dir_abs)
        candidates = sorted(
            (sub / "inpaint_out.mp4" for sub in out_root.iterdir()
             if sub.is_dir() and (sub / "inpaint_out.mp4").is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"ProPainter inpaint_out.mp4 not found under {out_root}."
            )
        from utils.video_io import extract_frames
        frames_out = extract_frames(str(candidates[0]))
        if not frames_out:
            raise RuntimeError(
                f"ProPainter produced an empty video at {candidates[0]}."
            )
        return frames_out


class FlowGuidedInpainter:
    """Fallback: flow-guided inpainting using optical flow warping + cv2.inpaint.

    Warps clean regions from neighboring frames to fill masked areas,
    then uses cv2.inpaint for remaining holes.
    """

    def __init__(self):
        pass

    def inpaint(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        output_dir: Optional[str] = None,
        search_range: int = 5,
    ) -> List[np.ndarray]:
        """Flow-guided inpainting with neighbor frame warping.

        Only borrows pixels from regions that are *unmasked* in the
        neighbor frame, avoiding copying object pixels into the hole.
        ``output_dir`` is ignored (API parity with ProPainterInpainter).
        """
        del output_dir
        results = []
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        for i in range(len(frames)):
            frame = frames[i].copy()
            mask = masks[i].copy()

            if mask.max() == 0:
                results.append(frame)
                continue

            filled = frame.copy()
            remaining_mask = mask.copy()

            for offset in range(1, search_range + 1):
                if remaining_mask.max() == 0:
                    break
                for j in [i - offset, i + offset]:
                    if j < 0 or j >= len(frames) or remaining_mask.max() == 0:
                        continue

                    warped = self._warp_frame(frames[j], grays[j], grays[i])

                    warped_nb_mask = self._warp_frame(
                        masks[j][:, :, np.newaxis] if masks[j].ndim == 2 else masks[j],
                        grays[j], grays[i],
                    )
                    if warped_nb_mask.ndim == 3:
                        warped_nb_mask = warped_nb_mask[:, :, 0]

                    need_fill = remaining_mask > 128
                    clean_in_neighbor = warped_nb_mask < 128
                    usable = need_fill & clean_in_neighbor

                    if usable.any():
                        usable_3ch = np.stack([usable] * 3, axis=-1)
                        filled[usable_3ch] = warped[usable_3ch]
                        remaining_mask[usable] = 0

            if remaining_mask.max() > 0:
                filled = cv2.inpaint(
                    filled, (remaining_mask > 128).astype(np.uint8),
                    inpaintRadius=5, flags=cv2.INPAINT_TELEA,
                )

            results.append(filled)
        return results

    @staticmethod
    def _warp_frame(
        src_frame: np.ndarray,
        src_gray: np.ndarray,
        dst_gray: np.ndarray,
    ) -> np.ndarray:
        """Warp src_frame to align with dst using dense optical flow."""
        flow = cv2.calcOpticalFlowFarneback(
            src_gray, dst_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        h, w = src_gray.shape
        coords = np.mgrid[0:h, 0:w].astype(np.float32)
        map_y = coords[0] + flow[..., 1]
        map_x = coords[1] + flow[..., 0]
        return cv2.remap(src_frame, map_x, map_y, cv2.INTER_LINEAR)


def get_inpainter(use_propainter: bool = True, **kwargs):
    """Factory: return ProPainter if available, else FlowGuidedInpainter."""
    if use_propainter and check_propainter_available():
        return ProPainterInpainter(**kwargs)
    print("ProPainter not available, using flow-guided fallback inpainter")
    return FlowGuidedInpainter()


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.video_io import extract_frames, frames_to_video, get_video_info
    from utils.mask_utils import load_masks

    parser = argparse.ArgumentParser(description="ProPainter / Flow-guided inpainting")
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", default="results/part2")
    parser.add_argument("--no-propainter", action="store_true")
    args = parser.parse_args()

    frames = extract_frames(args.video)
    masks = load_masks(args.masks)
    info = get_video_info(args.video)

    inpainter = get_inpainter(use_propainter=not args.no_propainter)
    inpainted = inpainter.inpaint(frames, masks)

    out_path = os.path.join(args.output, "inpainted.mp4")
    frames_to_video(inpainted, out_path, fps=info["fps"])
    print(f"Output: {out_path}")
