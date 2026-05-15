# AIAA 3201 Project 3 — Final Implementation Plan / Status

> Video Object Removal & Inpainting, Spring 2026

## Status Summary

Project status: **implemented and evaluated**.

All mandatory parts are complete:

- **Part 1 — Baseline:** YOLOv8-Seg + Lucas-Kanade sparse optical flow + temporal background propagation + OpenCV inpainting.
- **Part 2 — SOTA:** SAM 2 video tracking + ProPainter video inpainting.
- **Part 3 — Exploration:** conservative mask refinement + Stable Diffusion keyframe inpainting + optical-flow propagation + feathered/Poisson blending.
- **Mandatory data:** `bmx-trees`, `tennis`, and self-captured wild video processed.
- **Extra data:** DAVIS sequences processed: `crossing`, `car-shadow`, `blackswan`, `dance-jump`, `bus`.
- **Evaluation:** IoU / recall / PSNR / SSIM reports generated under `results/eval/`.
- **Visuals:** flowcharts and comparison grids generated under `report/figures/` and `results/`.
- **Report:** `report/main.tex` builds to `report/main.pdf` in CVPR-like two-column format.

## Implemented Roadmap

### Part 1: Hand-crafted baseline

Implemented files:

- `part1_baseline/detect_and_segment.py`
- `part1_baseline/optical_flow.py`
- `part1_baseline/inpaint_cv2.py`
- `part1_baseline/pipeline.py`

Core design:

1. Run YOLOv8-Seg per frame to detect dynamic object classes.
2. Use Lucas-Kanade sparse optical flow inside each connected component.
3. Filter static detections by motion magnitude and tracked-point reliability.
4. Keep recent masks under partial occlusion to avoid mask disappearance.
5. Dilate masks for motion blur.
6. Borrow clean background pixels from neighbor frames before using `cv2.inpaint` fallback.

Key lesson: pure spatial `cv2.inpaint` looked like mosaic/blur on large holes. Temporal propagation improved visible texture when nearby clean frames existed.

### Part 2: SOTA reproduction

Implemented files:

- `part2_sota/sam2_tracker.py`
- `part2_sota/propainter_inpaint.py`
- `part2_sota/pipeline.py`

Core design:

1. Generate initial proposals with YOLOv8-Seg.
2. Use SAM 2 video predictor for temporally consistent masks.
3. Use multi-frame initialization instead of frame-0-only seeding.
4. Run ProPainter with FP16 where possible.
5. Keep a fallback flow-guided inpainter; it avoids borrowing masked pixels from neighbor frames.

Key lesson: Part 2 gives the most stable perceptual results on most videos because segmentation and inpainting are both video-aware.

### Part 3: Exploration and optimization

Implemented files:

- `part3_exploration/sam3_upgrade.py`
- `part3_exploration/diffusion_inpaint.py`
- `part3_exploration/pipeline.py`

Core design:

1. SAM 3 is documented as unavailable in the current environment.
2. Fallback mask refinement uses light dilation and area-guarded GrabCut.
3. Stable Diffusion inpainting runs on selected keyframes only.
4. Non-keyframes are filled by optical-flow propagation between generated keyframes.
5. Hard mask replacement is replaced by Gaussian feathering and Poisson seamless cloning.

Key lesson: Part 3 improves masked PSNR on most sequences but may introduce temporal drift or hallucinated texture.

## Datasets Processed

Mandatory:

- `data/sample/bmx-trees`
- `data/sample/tennis`
- `data/wild_video/frames_480p`

Additional DAVIS:

- `crossing`
- `car-shadow`
- `blackswan`
- `dance-jump`
- `bus`

Outputs:

- `results/part1/<sequence>/inpainted.mp4`
- `results/part2/<sequence>/inpainted.mp4`
- `results/part3/<sequence>/inpainted.mp4`
- `results/eval/<sequence>.json`
- `results/comparison_<sequence>.png`

## Evaluation Summary

Metrics used:

- Mask quality: mean IoU and recall.
- Restoration quality: PSNR and SSIM, both full-frame and masked-region.

Important caveat: most videos do not provide clean-background ground truth, so PSNR/SSIM compare against original occluded frames. These metrics are useful for relative comparison but must be interpreted with qualitative grids.

Observed trends:

- Part 2 generally gives the best mask quality and temporal stability.
- Part 3 improves masked-region PSNR due to diffusion and seam-aware blending.
- Part 1 remains useful as a transparent baseline but struggles with large holes and complex occlusion.
- Shadows are not removed because masks and DAVIS annotations define object bodies, not cast shadows.

## Report Status

Report files:

- `report/main.tex`
- `report/main.pdf`
- `report/references.bib`
- `report/figures/`

Report includes:

- Abstract and introduction.
- Related work covering all 17 references in the project brief.
- Method section with flowcharts.
- Quantitative tables.
- Qualitative figures.
- Failure cases, limitations, and future work.

## Remaining Submission Tasks

Before final Canvas/GitHub submission:

1. Replace placeholder author names and emails in `report/main.tex`.
2. Replace placeholder GitHub URL in abstract.
3. Pack mandatory processed videos into `videos.zip`:
   - `bmx-trees` Part 1/2/3
   - `tennis` Part 1/2/3
   - `wild_video` Part 1/2/3
4. Ensure `.venv/`, full `data/`, large checkpoints, and temporary LaTeX files are not committed.
5. Push code to a public GitHub repository.

## Cleanup Notes

Safe to delete locally:

- `report/main.aux`
- `report/main.bbl`
- `report/main.blg`
- `report/main.log`

Keep empty `__init__.py` files; they mark Python packages.
