#!/bin/bash
# Part 1 Baseline Pipeline
# Usage: bash scripts/run_part1.sh <video_path>

set -e

VIDEO=${1:?"Usage: bash scripts/run_part1.sh <video_path>"}
OUTPUT="results/part1"

source .venv/bin/activate

echo "=== Part 1: Baseline Pipeline ==="
python part1_baseline/pipeline.py \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --model yolov8m-seg.pt \
    --conf 0.35 \
    --motion-threshold 2.0 \
    --inpaint-method telea

echo "=== Part 1 Complete ==="
echo "Results in: $OUTPUT"
