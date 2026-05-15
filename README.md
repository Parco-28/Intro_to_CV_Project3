# Video Object Removal & Inpainting

> AIAA 3201 — Introduction to Computer Vision, Project 3, Spring 2026

This repository implements a complete three-part video object removal and inpainting pipeline. Given a video containing dynamic foreground objects, the system detects/removes those objects and restores the missing regions with temporal information.

## Implemented Pipelines

| Part | Goal | Mask extraction | Inpainting |
|------|------|-----------------|------------|
| Part 1 — Baseline | Hand-crafted CV baseline | YOLOv8-Seg + Lucas-Kanade sparse optical flow | Temporal background propagation + `cv2.inpaint` fallback |
| Part 2 — SOTA | Academic reproduction | SAM 2 video tracking with multi-frame seeding | ProPainter video inpainting |
| Part 3 — Exploration | Optimization / extension | Conservative dilation + area-guarded GrabCut refinement | Stable Diffusion keyframes + optical-flow propagation + feathered/Poisson blending |

Part 2 gives the strongest temporal consistency in most videos. Part 3 improves masked-region PSNR on many sequences but may hallucinate texture or drift across frames.

## Results

Generated outputs are stored under `results/`:

```text
results/
├── part1/<sequence>/inpainted.mp4
├── part2/<sequence>/inpainted.mp4
├── part3/<sequence>/inpainted.mp4
├── eval/<sequence>.json
└── comparison_<sequence>.png
```

Processed sequences:

- Mandatory: `bmx-trees`, `tennis`, `wild_video`
- Additional DAVIS: `crossing`, `car-shadow`, `blackswan`, `dance-jump`, `bus`

Representative comparison figures:

- `results/comparison_bmx-trees.png`
- `results/comparison_tennis.png`
- `results/comparison_wild_video.png`
- `results/comparison_blackswan.png`
- `results/comparison_bus.png`
- `results/comparison_car-shadow.png`

## Environment

Tested setup:

- Windows 11
- Python 3.11
- NVIDIA RTX 4060 8GB
- CUDA-enabled PyTorch

### Install

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional / external modules:

```powershell
# SAM 2, if cloned under third_party/sam2
pip install -e third_party/sam2

# ProPainter dependencies, if cloned under third_party/ProPainter
pip install -r third_party/ProPainter/requirements.txt

# Stable Diffusion branch for Part 3
pip install diffusers transformers accelerate
```

Model checkpoints:

| Model | Location / behavior |
|-------|---------------------|
| YOLOv8-Seg | auto-downloaded by Ultralytics if missing |
| SAM 2.1 base+ | `third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt` |
| ProPainter | `third_party/ProPainter/weights/*.pth` |
| Stable Diffusion inpaint | Hugging Face cache; set `HF_ENDPOINT=https://hf-mirror.com` if needed |

Run environment check:

```powershell
python scripts/check_env.py
```

## Quick Start

Each pipeline accepts either an `.mp4` video or a directory of ordered frames.

### Run all parts

```powershell
.\scripts\run_all.ps1 -Source data\sample\bmx-trees
```

### Run individually

```powershell
python part1_baseline/pipeline.py --video data\sample\bmx-trees --output results\part1\bmx-trees
python part2_sota/pipeline.py --video data\sample\bmx-trees --output results\part2\bmx-trees
python part3_exploration/pipeline.py --video data\sample\bmx-trees --masks results\part2\bmx-trees\masks --output results\part3\bmx-trees
```

For Part 3 with fallback inpainting only:

```powershell
python part3_exploration/pipeline.py --video data\sample\bmx-trees --masks results\part2\bmx-trees\masks --output results\part3\bmx-trees --no-diffusion
```

## Evaluation

Use `evaluation/run_eval.py` for full Part 1/2/3 comparison.

Sample data:

```powershell
python evaluation\run_eval.py --sample bmx-trees --out results\eval\bmx-trees.json --save-grid results\comparison_bmx-trees.png
python evaluation\run_eval.py --sample tennis --out results\eval\tennis.json --save-grid results\comparison_tennis.png
```

DAVIS data:

```powershell
python evaluation\run_eval.py --sample car-shadow `
  --data-root data\davis\DAVIS\JPEGImages\480p `
  --gt-mask-dir data\davis\DAVIS\Annotations\480p\car-shadow `
  --out results\eval\car-shadow.json `
  --save-grid results\comparison_car-shadow.png
```

Run multiple DAVIS sequences:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts\run_davis_sequences.py --sequences car-shadow blackswan dance-jump bus
```

## Report

Report source:

```text
report/main.tex
report/references.bib
report/figures/
```

Build:

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Current report uses a CVPR-like two-column article format because the official `cvpr.sty` file is not included in this repository. It uses Times-style fonts via `newtxtext,newtxmath` and includes all required sections from the project brief.

## Project Structure

```text
part1_baseline/        # YOLOv8-Seg + LK flow + temporal propagation + cv2 fallback
part2_sota/            # SAM 2 tracker + ProPainter wrapper + fallback inpainter
part3_exploration/     # mask refinement + Stable Diffusion keyframe inpainting
evaluation/            # metrics, evaluation runner, visualization
utils/                 # video/frame I/O and mask I/O
scripts/               # run scripts, DAVIS batch runner, flowchart generation, utilities
report/                # LaTeX report and figures
results/               # generated videos, metrics, comparison images
data/                  # local input datasets, not intended for GitHub upload
third_party/           # local external repositories (SAM 2, ProPainter)
```

## Known Limitations

- Shadows are usually not removed because SAM 2 and DAVIS annotations segment object bodies, not cast shadows.
- Stable Diffusion can hallucinate plausible but temporally inconsistent texture.
- Wild-video metrics are limited because no clean background or ground-truth masks exist.
- 8GB GPU memory limits resolution; the wild video was resized to 480p frames.

## Submission Notes

Before final submission:

1. Pack mandatory demo videos into `videos.zip` (done — 36.6 MB, 24 mp4s).
2. Do not commit `.venv/`, full `data/`, or large checkpoint caches.
3. Keep `report/main.pdf`, key comparison PNGs, and clear usage instructions.
4. GitHub repo: https://github.com/Parco-28/Intro_to_CV_Project3
