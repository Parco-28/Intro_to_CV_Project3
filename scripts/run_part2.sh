#!/bin/bash
# Part 2 SOTA Pipeline
# Usage: bash scripts/run_part2.sh <video_path>

set -e

VIDEO=${1:?"Usage: bash scripts/run_part2.sh <video_path>"}
OUTPUT="results/part2"

source .venv/bin/activate

echo "=== Part 2: SOTA Pipeline ==="
python part2_sota/pipeline.py \
    --video "$VIDEO" \
    --output "$OUTPUT"

echo "=== Part 2 Complete ==="
echo "Results in: $OUTPUT"
