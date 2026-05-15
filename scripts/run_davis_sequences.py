"""Run Part 1 / 2 / 3 on multiple DAVIS 2017 trainval-480p sequences.

Usage (from repo root, venv activated):
    set HF_ENDPOINT=https://hf-mirror.com   # optional, Windows
    python scripts/run_davis_sequences.py --sequences car-shadow blackswan dance-jump bus

Defaults run four short/medium sequences (excluding crossing if already done).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JPEG = ROOT / "data" / "davis" / "DAVIS" / "JPEGImages" / "480p"


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT))
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sequences",
        nargs="+",
        default=["car-shadow", "blackswan", "dance-jump", "bus"],
        help="DAVIS sequence folder names under JPEGImages/480p",
    )
    ap.add_argument("--skip-part3", action="store_true", help="Only Part 1 and Part 2")
    ap.add_argument("--no-diffusion", action="store_true", help="Part 3 uses fallback inpainter (faster)")
    args = ap.parse_args()

    py = sys.executable
    failures: list[str] = []

    for name in args.sequences:
        vid = JPEG / name
        if not vid.is_dir():
            print(f"[skip] missing frames dir: {vid}")
            failures.append(f"{name}: no frames")
            continue

        p1 = run([py, "part1_baseline/pipeline.py", "--video", str(vid), "--output", f"results/part1/{name}"])
        if p1 != 0:
            failures.append(f"{name}: part1 exit {p1}")
            continue

        p2 = run([py, "part2_sota/pipeline.py", "--video", str(vid), "--output", f"results/part2/{name}"])
        if p2 != 0:
            failures.append(f"{name}: part2 exit {p2}")
            continue

        if args.skip_part3:
            continue

        cmd = [
            py, "part3_exploration/pipeline.py",
            "--video", str(vid),
            "--masks", f"results/part2/{name}/masks",
            "--output", f"results/part3/{name}",
        ]
        if args.no_diffusion:
            cmd.append("--no-diffusion")
        p3 = run(cmd)
        if p3 != 0:
            failures.append(f"{name}: part3 exit {p3}")

    if failures:
        print("\n--- Failures ---", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1
    print("\nAll sequences finished OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
