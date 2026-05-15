"""Convert a directory of frames into an MP4 video.

Useful for preparing the mandatory ``videos.zip`` submission once the
pipelines have written their inpainted frame sequences.

Example
-------
python scripts/frames_to_mp4.py results/part2/bmx-trees/frames out/bmx_part2.mp4 --fps 24
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.video_io import load_frames_from_dir, frames_to_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Frame directory -> MP4")
    parser.add_argument("input", help="Directory containing frame images")
    parser.add_argument("output", help="Output .mp4 path")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--codec", default="mp4v", choices=["mp4v", "avc1", "h264"])
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_dir():
        print(f"Not a directory: {in_path}")
        return 1

    frames = load_frames_from_dir(str(in_path))
    if not frames:
        print(f"No frames found in {in_path}")
        return 1
    print(f"Loaded {len(frames)} frames from {in_path}")

    frames_to_video(frames, args.output, fps=args.fps, codec=args.codec)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
