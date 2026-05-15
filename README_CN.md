# 视频动态目标移除与修复

> AIAA 3201 — 计算机视觉导论，Project 3，2026 春季学期

本仓库实现了完整的三阶段视频动态目标移除与背景修复流程。输入一段包含动态前景目标的视频，系统会先检测并移除目标，再利用时序信息修复缺失区域。

## 已实现方案

| 部分 | 目标 | 掩码提取 | 修复方式 |
|------|------|---------|---------|
| Part 1 — 基线 | 手工 CV 基线 | YOLOv8-Seg + Lucas-Kanade 稀疏光流 | 时序背景传播 + `cv2.inpaint` 兜底 |
| Part 2 — SOTA | 论文复现 | SAM 2 视频跟踪，多帧初始化 | ProPainter 视频修复 |
| Part 3 — 探索 | 优化 / 扩展 | 轻量膨胀 + 面积约束 GrabCut 精炼 | Stable Diffusion keyframe + 光流传播 + feather/Poisson 融合 |

一般情况下，Part 2 的时序稳定性最好；Part 3 往往能提高 masked-region PSNR，但可能引入纹理幻觉或帧间漂移。

## 结果输出

所有结果都在 `results/` 下：

```text
results/
├── part1/<sequence>/inpainted.mp4
├── part2/<sequence>/inpainted.mp4
├── part3/<sequence>/inpainted.mp4
├── eval/<sequence>.json
└── comparison_<sequence>.png
```

已处理序列：

- 必做：`bmx-trees`、`tennis`、`wild_video`
- 额外 DAVIS：`crossing`、`car-shadow`、`blackswan`、`dance-jump`、`bus`

代表性对比图：

- `results/comparison_bmx-trees.png`
- `results/comparison_tennis.png`
- `results/comparison_wild_video.png`
- `results/comparison_blackswan.png`
- `results/comparison_bus.png`
- `results/comparison_car-shadow.png`

## 环境

测试环境：

- Windows 11
- Python 3.11
- NVIDIA RTX 4060 8GB
- CUDA PyTorch

### 安装

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

可选 / 外部模块：

```powershell
# SAM 2（若已克隆到 third_party/sam2）
pip install -e third_party/sam2

# ProPainter 依赖（若已克隆到 third_party/ProPainter）
pip install -r third_party/ProPainter/requirements.txt

# Part 3 的 Stable Diffusion 分支
pip install diffusers transformers accelerate
```

模型权重：

| 模型 | 位置 / 行为 |
|------|-------------|
| YOLOv8-Seg | 若缺失会由 Ultralytics 自动下载 |
| SAM 2.1 base+ | `third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt` |
| ProPainter | `third_party/ProPainter/weights/*.pth` |
| Stable Diffusion inpaint | Hugging Face 缓存；必要时设置 `HF_ENDPOINT=https://hf-mirror.com` |

环境检查：

```powershell
python scripts\check_env.py
```

## 快速开始

每个 pipeline 都接受 `.mp4` 或按顺序排列的帧目录。

### 一键运行全部

```powershell
.\scripts\run_all.ps1 -Source data\sample\bmx-trees
```

### 单独运行

```powershell
python part1_baseline/pipeline.py --video data\sample\bmx-trees --output results\part1\bmx-trees
python part2_sota/pipeline.py --video data\sample\bmx-trees --output results\part2\bmx-trees
python part3_exploration/pipeline.py --video data\sample\bmx-trees --masks results\part2\bmx-trees\masks --output results\part3\bmx-trees
```

Part 3 仅用 fallback 修复：

```powershell
python part3_exploration/pipeline.py --video data\sample\bmx-trees --masks results\part2\bmx-trees\masks --output results\part3\bmx-trees --no-diffusion
```

## 评估

使用 `evaluation/run_eval.py` 生成 Part 1/2/3 对比评估。

Sample 数据：

```powershell
python evaluation\run_eval.py --sample bmx-trees --out results\eval\bmx-trees.json --save-grid results\comparison_bmx-trees.png
python evaluation\run_eval.py --sample tennis --out results\eval\tennis.json --save-grid results\comparison_tennis.png
```

DAVIS 数据：

```powershell
python evaluation\run_eval.py --sample car-shadow `
  --data-root data\davis\DAVIS\JPEGImages\480p `
  --gt-mask-dir data\davis\DAVIS\Annotations\480p\car-shadow `
  --out results\eval\car-shadow.json `
  --save-grid results\comparison_car-shadow.png
```

批量运行 DAVIS：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
python scripts\run_davis_sequences.py --sequences car-shadow blackswan dance-jump bus
```

## 报告

报告源码：

```text
report/main.tex
report/references.bib
report/figures/
```

编译：

```powershell
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

当前报告使用的是 **CVPR 风格的双栏 article 版式**。仓库里没有官方 `cvpr.sty`，所以采用兼容的双栏排版并使用 Times 风格字体 `newtxtext,newtxmath`。

## 项目结构

```text
part1_baseline/        # YOLOv8-Seg + LK flow + 时序传播 + cv2 兜底
part2_sota/            # SAM 2 跟踪 + ProPainter 封装 + fallback 修复
part3_exploration/     # 掩码精炼 + Stable Diffusion keyframe 修复
evaluation/            # 指标计算、评估脚本、可视化
utils/                 # 视频/帧 I/O、掩码 I/O
scripts/               # 运行脚本、DAVIS 批处理、流程图、工具
report/                # LaTeX 报告与图表
results/               # 生成的视频、指标、对比图
data/                  # 本地输入数据，不建议上传 GitHub
third_party/           # 外部仓库（SAM 2、ProPainter）
```

## 已知限制

- 阴影通常不会被去除，因为 SAM 2 / DAVIS 标注的是物体本体，不包含 cast shadow。
- Stable Diffusion 可能生成看起来自然但帧间不一致的纹理。
- wild video 没有 clean background 或 GT mask，因此只能做有限的定量评估。
- 8GB GPU 限制分辨率；wild video 已降到 480p 帧。

## 提交前提醒

1. 替换 `report/main.tex` 中的作者信息和 GitHub 链接。
2. 打包必做视频为 `videos.zip`。
3. 不要提交 `.venv/`、完整 `data/`、大模型缓存。
4. 保留 `report/main.pdf`、关键对比图、清晰的运行说明。
