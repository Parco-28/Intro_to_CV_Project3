"""Part 3 — SAM 3 mask upgrade module.

Attempts to use SAM 3 for higher-quality segmentation masks.
Falls back to morphological refinement + GrabCut if SAM 3 is unavailable.

SAM 3 (arXiv 2511.16719) extends SAM with concept-level segmentation.
When available, we use it as a refinement pass on existing masks to
obtain pixel-precise boundaries.  When unavailable, morphological
cleanup + GrabCut edge-aware refinement provides a reasonable
approximation.
"""

import cv2
import numpy as np
from typing import List, Optional
from pathlib import Path


def check_sam3_available() -> bool:
    """Check if SAM 3 is available (``sam3`` package)."""
    try:
        from sam3 import Sam3Predictor  # noqa: F401
        return True
    except ImportError:
        return False


class SAM3Refiner:
    """Refine masks using SAM 3's concept-aware segmentation.

    Uses existing masks as spatial prompts to SAM 3 for boundary
    refinement — the "Mask Upgrade" direction from the project spec.
    """

    def __init__(self):
        from sam3 import Sam3Predictor, sam3_model_registry
        self.predictor = Sam3Predictor(sam3_model_registry["default"]())
        self.predictor.model.eval()
        print("SAM 3 model loaded for mask refinement")

    def refine_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return mask
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)
        refined, _, _ = self.predictor.predict(
            mask_input=mask[None, :, :].astype(np.float32) / 255.0,
            multimask_output=False,
        )
        return (refined[0] > 0.5).astype(np.uint8) * 255

    def refine_masks(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        return [self.refine_mask(f, m) for f, m in zip(frames, masks)]


class MaskRefiner:
    """Fallback refiner using morphological operations and GrabCut.

    This is the degraded path when SAM 3 is not installed.
    """

    def __init__(self, use_grabcut: bool = True):
        self.use_grabcut = use_grabcut

    def refine_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return mask

        orig_area = np.count_nonzero(mask)

        # Light dilation only (cover motion blur edges)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.dilate(mask, kernel_dilate, iterations=1)

        if self.use_grabcut:
            candidate = self._grabcut_refine(frame, refined)
            cand_area = np.count_nonzero(candidate)
            if orig_area > 0 and 0.90 < cand_area / orig_area < 1.10:
                refined = candidate

        return refined

    def refine_masks(self, frames: List[np.ndarray], masks: List[np.ndarray]) -> List[np.ndarray]:
        return [self.refine_mask(f, m) for f, m in zip(frames, masks)]

    @staticmethod
    def _morphological_refine(mask: np.ndarray) -> np.ndarray:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(refined)
        min_area = refined.shape[0] * refined.shape[1] * 0.001
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(clean, [cnt], -1, 255, -1)

        return clean

    @staticmethod
    def _grabcut_refine(frame: np.ndarray, mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        try:
            gc_mask = np.where(mask > 128, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            sure_fg = cv2.erode(mask, kernel, iterations=2)
            sure_bg_inv = cv2.dilate(mask, kernel, iterations=2)

            gc_mask[sure_fg > 128] = cv2.GC_FGD
            gc_mask[sure_bg_inv == 0] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(frame, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

            result = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            return result
        except cv2.error:
            return mask


def get_mask_refiner(**kwargs):
    """Return SAM3Refiner if available, else fallback MaskRefiner.

    The fallback uses morphological cleanup + GrabCut, which approximates
    SAM 3's boundary refinement but lacks its learned semantic priors.
    """
    if check_sam3_available():
        print("SAM 3 available — using SAM 3 for mask refinement")
        try:
            return SAM3Refiner()
        except Exception as e:
            print(f"SAM 3 init failed ({e}), falling back to morphological + GrabCut")
    else:
        print(
            "SAM 3 not available — using morphological + GrabCut refinement.\n"
            "  Reason: 'sam3' package not installed (arXiv:2511.16719).\n"
            "  Install: pip install sam3  (when released)"
        )
    return MaskRefiner(**kwargs)
