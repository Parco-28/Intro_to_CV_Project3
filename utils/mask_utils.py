"""Mask utilities: I/O, morphological ops, and helpers."""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Optional


def imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    """``cv2.imread`` replacement that handles non-ASCII paths on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """``cv2.imwrite`` replacement that handles non-ASCII paths on Windows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def save_masks(masks: List[np.ndarray], output_dir: str) -> None:
    """Save binary masks as grayscale PNGs (unicode-safe)."""
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(masks):
        imwrite_unicode(os.path.join(output_dir, f"{i:05d}.png"), mask)


def load_masks(mask_dir: str) -> List[np.ndarray]:
    """Load masks from a directory, sorted by filename (unicode-safe)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        p for p in Path(mask_dir).iterdir() if p.suffix.lower() in exts
    )
    out = []
    for p in paths:
        img = imread_unicode(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out.append(img)
    return out


def dilate_mask(mask: np.ndarray, kernel_size: int = 15, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask to cover object edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def erode_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Erode a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Combine multiple binary masks into one via logical OR."""
    if not masks:
        raise ValueError("Empty mask list")
    combined = masks[0].copy()
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return combined


def binarize_mask(mask: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Ensure mask is strictly binary (0 or 255)."""
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def davis_palette_to_binary(mask_path: str, instance_id: Optional[int] = None) -> np.ndarray:
    """Convert a DAVIS-style indexed PNG (palette PNG, 0 = background) to a binary mask.

    Args:
        mask_path: Path to the ``Annotations/480p/<video>/<frame>.png`` file.
        instance_id: If provided, only that instance is returned; otherwise the
            union of all instances (anything > 0) is returned.

    Returns:
        ``np.uint8`` array of shape (H, W) with values in {0, 255}.
    """
    img = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(mask_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if instance_id is not None:
        return ((img == instance_id).astype(np.uint8)) * 255
    return ((img > 0).astype(np.uint8)) * 255


def load_davis_masks(mask_dir: str, instance_id: Optional[int] = None) -> List[np.ndarray]:
    """Load all DAVIS annotation PNGs in ``mask_dir`` as binary masks."""
    exts = {".png"}
    paths = sorted(
        p for p in Path(mask_dir).iterdir() if p.suffix.lower() in exts
    )
    return [davis_palette_to_binary(str(p), instance_id=instance_id) for p in paths]
