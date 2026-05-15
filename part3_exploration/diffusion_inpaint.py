"""Part 3 — Diffusion-based generative inpainting.

Uses Stable Diffusion + ControlNet for high-quality generative video inpainting.
Falls back to enhanced multi-scale cv2 inpainting if diffusion models are unavailable.
"""

import cv2
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path


def check_diffusion_available() -> bool:
    """Check if diffusers library is installed."""
    try:
        import diffusers
        return True
    except ImportError:
        return False


class DiffusionInpainter:
    """Stable Diffusion inpainting pipeline for video frames."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None

    def load_model(self):
        """Load the SD inpainting pipeline (prefer local cache)."""
        import os
        from diffusers import StableDiffusionInpaintPipeline

        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        kwargs = dict(
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id, local_files_only=True, **kwargs
            ).to(self.device)
        except Exception:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id, **kwargs
            ).to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        print(f"Diffusion inpainting model loaded on {self.device}")

    def inpaint_frame(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prompt: str = "clean background, no objects, natural scene",
        negative_prompt: str = "person, people, car, vehicle, animal, object",
        strength: float = 0.75,
        guidance_scale: float = 7.5,
    ) -> np.ndarray:
        """Inpaint a single frame using Stable Diffusion."""
        if self.pipe is None:
            self.load_model()

        from PIL import Image

        h, w = frame.shape[:2]
        target_h = (h // 8) * 8
        target_w = (w // 8) * 8

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((target_w, target_h))
        mask_pil = Image.fromarray(mask).resize((target_w, target_h))

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=frame_pil,
            mask_image=mask_pil,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=30,
        ).images[0]

        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if result_np.shape[:2] != (h, w):
            result_np = cv2.resize(result_np, (w, h))

        output = self._blend_with_feather(frame, result_np, mask)
        return output

    @staticmethod
    def _blend_with_feather(
        bg: np.ndarray, fg: np.ndarray, mask: np.ndarray, feather_px: int = 12
    ) -> np.ndarray:
        """Feathered alpha blend + Poisson fallback for seamless edges."""
        alpha = cv2.GaussianBlur(
            (mask > 128).astype(np.float32), (feather_px * 2 + 1, feather_px * 2 + 1), 0
        )
        alpha = alpha[..., None]
        blended = (alpha * fg.astype(np.float32) + (1 - alpha) * bg.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        try:
            center = tuple(np.flip(np.array(np.where(mask > 128)).mean(axis=1).astype(int)))
            poisson = cv2.seamlessClone(fg, bg, mask, center, cv2.NORMAL_CLONE)
            inner = cv2.erode(mask, np.ones((feather_px, feather_px), np.uint8))
            inner_3 = np.stack([inner > 128] * 3, axis=-1)
            blended[inner_3] = poisson[inner_3]
        except cv2.error:
            pass
        return blended

    def inpaint(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        prompt: str = "clean background, no objects, natural scene",
        keyframe_interval: int = 10,
    ) -> List[np.ndarray]:
        """Inpaint using keyframe diffusion + temporal propagation.

        Runs SD only on keyframes (every *keyframe_interval* frames among
        those that need inpainting).  Non-keyframes are filled by warping
        the nearest keyframe result via optical flow, then blending.
        """
        from tqdm import tqdm

        n = len(frames)
        needs_inpaint = [masks[i].max() > 0 for i in range(n)]

        keyframe_results: dict = {}
        keyframe_indices = []
        for i in range(n):
            if needs_inpaint[i] and (
                len(keyframe_indices) == 0
                or i - keyframe_indices[-1] >= keyframe_interval
            ):
                keyframe_indices.append(i)

        if keyframe_indices and keyframe_indices[-1] != n - 1:
            last_needed = max(i for i in range(n) if needs_inpaint[i])
            if last_needed not in keyframe_indices:
                keyframe_indices.append(last_needed)

        print(f"  Diffusion keyframes: {len(keyframe_indices)} / {sum(needs_inpaint)} masked frames")
        for idx in tqdm(keyframe_indices, desc="Diffusion keyframes"):
            keyframe_results[idx] = self.inpaint_frame(frames[idx], masks[idx], prompt=prompt)

        if not keyframe_indices:
            return list(frames)

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        results = [None] * n

        for i in range(n):
            if not needs_inpaint[i]:
                results[i] = frames[i]
            elif i in keyframe_results:
                results[i] = keyframe_results[i]
            else:
                prev_kf = max((k for k in keyframe_indices if k <= i), default=None)
                next_kf = min((k for k in keyframe_indices if k >= i), default=None)

                if prev_kf is not None and next_kf is not None and prev_kf != next_kf:
                    w_prev = (next_kf - i) / (next_kf - prev_kf)
                    warped_prev = self._warp_result(keyframe_results[prev_kf], grays[prev_kf], grays[i])
                    warped_next = self._warp_result(keyframe_results[next_kf], grays[next_kf], grays[i])
                    blended = (w_prev * warped_prev.astype(np.float64)
                               + (1 - w_prev) * warped_next.astype(np.float64))
                    propagated = np.clip(blended, 0, 255).astype(np.uint8)
                elif prev_kf is not None:
                    propagated = self._warp_result(keyframe_results[prev_kf], grays[prev_kf], grays[i])
                else:
                    propagated = self._warp_result(keyframe_results[next_kf], grays[next_kf], grays[i])

                mask_3ch = np.stack([masks[i] > 128] * 3, axis=-1)
                out = frames[i].copy()
                out[mask_3ch] = propagated[mask_3ch]
                results[i] = self._blend_with_feather(frames[i], propagated, masks[i])

        return results

    @staticmethod
    def _warp_result(src: np.ndarray, src_gray: np.ndarray, dst_gray: np.ndarray) -> np.ndarray:
        """Warp an inpainted result to align with a different frame."""
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


class EnhancedInpainter:
    """Fallback: multi-scale inpainting with temporal blending.

    Combines multi-resolution cv2.inpaint with neighbor-frame warping
    for better quality than simple single-pass inpainting.
    """

    def inpaint(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Multi-scale inpainting with temporal consistency."""
        results = []
        for i in range(len(frames)):
            if masks[i].max() == 0:
                results.append(frames[i])
                continue
            result = self._multiscale_inpaint(frames[i], masks[i])
            results.append(result)

        return self._temporal_blend(results, masks)

    @staticmethod
    def _multiscale_inpaint(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint at multiple scales and blend for better quality."""
        scales = [1.0, 0.5, 0.25]
        h, w = frame.shape[:2]
        accumulated = np.zeros_like(frame, dtype=np.float64)
        weight_sum = 0.0

        for scale in scales:
            sh, sw = int(h * scale), int(w * scale)
            if sh < 16 or sw < 16:
                continue
            small_frame = cv2.resize(frame, (sw, sh))
            small_mask = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            small_mask_bin = (small_mask > 128).astype(np.uint8)

            inpainted = cv2.inpaint(small_frame, small_mask_bin, 7, cv2.INPAINT_TELEA)
            upscaled = cv2.resize(inpainted, (w, h))

            weight = 1.0 / scale
            accumulated += upscaled.astype(np.float64) * weight
            weight_sum += weight

        blended = (accumulated / weight_sum).astype(np.uint8)

        mask_3ch = np.stack([mask > 128] * 3, axis=-1)
        output = frame.copy()
        output[mask_3ch] = blended[mask_3ch]
        return output

    @staticmethod
    def _temporal_blend(
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        alpha: float = 0.4,
    ) -> List[np.ndarray]:
        """EMA temporal smoothing on inpainted mask regions."""
        blended = [frames[0].copy()]
        for i in range(1, len(frames)):
            if masks[i].max() == 0:
                blended.append(frames[i])
                continue
            feather = cv2.GaussianBlur(
                (masks[i] > 128).astype(np.float32), (11, 11), 0
            )[..., None]
            prev = blended[i - 1].astype(np.float32)
            curr = frames[i].astype(np.float32)
            mixed = alpha * prev + (1 - alpha) * curr
            result = (feather * mixed + (1 - feather) * curr)
            blended.append(np.clip(result, 0, 255).astype(np.uint8))
        return blended


def get_diffusion_inpainter(use_diffusion: bool = True, **kwargs):
    """Factory: return DiffusionInpainter if available, else EnhancedInpainter."""
    if use_diffusion and check_diffusion_available():
        return DiffusionInpainter(**kwargs)
    print("Using enhanced multi-scale fallback inpainter")
    return EnhancedInpainter()
