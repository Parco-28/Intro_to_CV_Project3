#!/bin/bash
# Part 3 Exploration Pipeline
# Usage: bash scripts/run_part3.sh <video_path>

set -e

VIDEO=${1:?"Usage: bash scripts/run_part3.sh <video_path>"}
MASKS="results/part2/masks"
OUTPUT="results/part3"

source .venv/bin/activate

echo "=== Part 3: Exploration Pipeline ==="
python part3_exploration/pipeline.py \
    --video "$VIDEO" \
    --masks "$MASKS" \
    --output "$OUTPUT"

echo "=== Part 3 Complete ==="
echo "Results in: $OUTPUT"
