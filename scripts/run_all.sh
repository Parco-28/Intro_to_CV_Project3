#!/bin/bash
# Run all three parts sequentially
# Usage: bash scripts/run_all.sh <video_path>

set -e

VIDEO=${1:?"Usage: bash scripts/run_all.sh <video_path>"}

echo "=========================================="
echo "  Video Object Removal & Inpainting"
echo "=========================================="

bash scripts/run_part1.sh "$VIDEO"
echo ""
bash scripts/run_part2.sh "$VIDEO"
echo ""
bash scripts/run_part3.sh "$VIDEO"

echo ""
echo "=========================================="
echo "  All parts complete!"
echo "  Results in: results/"
echo "=========================================="
