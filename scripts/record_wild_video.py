"""Record a short ``wild`` video from the default webcam.

The project rubric requires every group to submit at least one self-shot
clip that contains a dynamic object. Use this script as a fallback when
filming on a phone is inconvenient.

Example
-------
python scripts/record_wild_video.py data/wild_video/clip.mp4 --duration 10 --fps 24
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a clip from the webcam")
    parser.add_argument("output", help="Output .mp4 path")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0, help="Seconds")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (w, h))

    total = int(args.duration * args.fps)
    print(f"Recording {total} frames at {w}x{h}@{args.fps}fps -> {out_path}")
    print("Press 'q' to stop early. Make sure a moving object is in frame.")

    start = time.time()
    written = 0
    try:
        while written < total:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1

            preview = frame.copy()
            elapsed = time.time() - start
            cv2.putText(
                preview, f"{elapsed:4.1f}s  {written}/{total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )
            cv2.imshow("Recording (q to stop)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved {written} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
