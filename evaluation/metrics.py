"""Evaluation metrics: IoU, PSNR, SSIM for mask and inpainting quality."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def _match_mask_size(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Resize a predicted mask to match the GT mask's spatial size."""
    if pred.shape[:2] != gt.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    pred_bin = pred > 128
    gt_bin = gt > 128
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_iou_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute recall: what fraction of GT mask is captured by prediction."""
    pred_bin = pred > 128
    gt_bin = gt > 128
    gt_sum = gt_bin.sum()
    if gt_sum == 0:
        return 1.0
    return float(np.logical_and(pred_bin, gt_bin).sum() / gt_sum)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    return float(psnr(img1, img2, data_range=255))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images."""
    if img1.ndim == 3:
        return float(ssim(img1, img2, channel_axis=2, data_range=255))
    return float(ssim(img1, img2, data_range=255))


def evaluate_masks(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """Evaluate mask quality across a sequence."""
    ious, recalls = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        pred = _match_mask_size(pred, gt)
        ious.append(compute_iou(pred, gt))
        recalls.append(compute_iou_recall(pred, gt))
    return {
        "iou_mean": float(np.mean(ious)) if ious else 0.0,
        "iou_std": float(np.std(ious)) if ious else 0.0,
        "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recall_std": float(np.std(recalls)) if recalls else 0.0,
        "n_frames": len(ious),
    }


def _match_size(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize ``a`` to match ``b``'s spatial size if they differ."""
    if a.shape[:2] != b.shape[:2]:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
    return a


def evaluate_inpainting(
    inpainted: List[np.ndarray],
    ground_truth: List[np.ndarray],
    masks: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Evaluate inpainting quality. PSNR / SSIM are reported both for the
    full frame and (if masks are provided) restricted to the masked regions."""
    psnrs_full, ssims_full = [], []
    psnrs_mask, ssims_mask = [], []

    for i in range(min(len(inpainted), len(ground_truth))):
        inp = inpainted[i]
        gt = _match_size(ground_truth[i], inp)

        psnrs_full.append(compute_psnr(gt, inp))
        ssims_full.append(compute_ssim(gt, inp))

        if masks is not None and i < len(masks) and masks[i] is not None and masks[i].max() > 0:
            m = _match_size(masks[i], inp)
            mask_bool = m > 128
            if mask_bool.sum() < 100:
                continue
            mask_3 = np.stack([mask_bool] * 3, axis=-1) if inp.ndim == 3 else mask_bool

            inp_region = inp[mask_3].reshape(-1).astype(np.float64)
            gt_region = gt[mask_3].reshape(-1).astype(np.float64)
            mse = float(np.mean((inp_region - gt_region) ** 2))
            if mse > 0:
                p = 10 * np.log10((255.0 ** 2) / mse)
                psnrs_mask.append(float(p))

            ys, xs = np.where(mask_bool)
            if len(ys) > 0:
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                if (y1 - y0) >= 7 and (x1 - x0) >= 7:
                    crop_inp = inp[y0:y1, x0:x1]
                    crop_gt = gt[y0:y1, x0:x1]
                    ssims_mask.append(compute_ssim(crop_gt, crop_inp))

    def stats(arr: List[float], key: str) -> Dict[str, float]:
        return {
            f"{key}_mean": float(np.mean(arr)) if arr else 0.0,
            f"{key}_std": float(np.std(arr)) if arr else 0.0,
        }

    out = {}
    out.update(stats(psnrs_full, "psnr"))
    out.update(stats(ssims_full, "ssim"))
    out.update(stats(psnrs_mask, "psnr_masked"))
    out.update(stats(ssims_mask, "ssim_masked"))
    return out


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'=' * 40}")
    print(f"  {title}")
    print(f"{'=' * 40}")
    for k, v in metrics.items():
        print(f"  {k:>15s}: {v:.4f}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from utils.mask_utils import load_masks
    from utils.video_io import load_frames_from_dir

    parser = argparse.ArgumentParser(description="Evaluate masks or inpainting")
    parser.add_argument("--pred-masks", help="Predicted masks dir")
    parser.add_argument("--gt-masks", help="Ground truth masks dir")
    parser.add_argument("--inpainted", help="Inpainted frames dir")
    parser.add_argument("--ground-truth", help="Ground truth frames dir")
    args = parser.parse_args()

    if args.pred_masks and args.gt_masks:
        pred = load_masks(args.pred_masks)
        gt = load_masks(args.gt_masks)
        metrics = evaluate_masks(pred, gt)
        print_metrics(metrics, "Mask Evaluation")

    if args.inpainted and args.ground_truth:
        inp = load_frames_from_dir(args.inpainted)
        gt = load_frames_from_dir(args.ground_truth)
        metrics = evaluate_inpainting(inp, gt)
        print_metrics(metrics, "Inpainting Evaluation")
