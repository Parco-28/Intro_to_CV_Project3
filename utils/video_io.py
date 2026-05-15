"""Video I/O utilities: frame extraction and video assembly.

All file-system access is unicode-safe so that paths containing CJK
characters (very common on the user's machine) round-trip cleanly through
OpenCV, which otherwise silently fails on Windows.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def _imread_unicode(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _imwrite_unicode(path: str, img: np.ndarray, params: Optional[List[int]] = None) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def _videocapture_unicode(video_path: str) -> cv2.VideoCapture:
    """Open a VideoCapture with an FFmpeg-style fallback for unicode paths."""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)


def extract_frames(video_path: str, output_dir: Optional[str] = None) -> List[np.ndarray]:
    """Extract all frames from a video file.

    Returns list of BGR frames. Optionally saves them to output_dir.
    """
    cap = _videocapture_unicode(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if output_dir:
            _imwrite_unicode(os.path.join(output_dir, f"{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    return frames


def get_video_info(video_path: str) -> dict:
    """Return fps, width, height, frame_count for a video."""
    cap = _videocapture_unicode(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> None:
    """Write a list of BGR frames to a video file (unicode-safe)."""
    if not frames:
        raise ValueError("Empty frame list")
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        writer.release()
        # Fallback: write to a temp ASCII path then move
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(output_path).suffix)
        tmp.close()
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            raise IOError(f"Cannot open VideoWriter for {output_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        shutil.move(tmp.name, output_path)
        return

    for frame in frames:
        writer.write(frame)
    writer.release()


def load_frames_from_dir(frame_dir: str) -> List[np.ndarray]:
    """Load frames from a directory of images, sorted by filename (unicode-safe)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        p for p in Path(frame_dir).iterdir() if p.suffix.lower() in exts
    )
    out = []
    for p in paths:
        img = _imread_unicode(str(p))
        if img is not None:
            out.append(img)
    return out


def load_video_or_frames(path: str) -> List[np.ndarray]:
    """Auto-detect: extract frames from a video file or load a frame directory."""
    p = Path(path)
    if p.is_dir():
        return load_frames_from_dir(str(p))
    if p.is_file():
        return extract_frames(str(p))
    raise FileNotFoundError(f"Not a file or directory: {path}")


def get_source_info(path: str, default_fps: float = 24.0) -> dict:
    """Return (fps, width, height, frame_count) for a video file or frame dir."""
    p = Path(path)
    if p.is_file():
        return get_video_info(str(p))
    if p.is_dir():
        frames = load_frames_from_dir(str(p))
        if not frames:
            raise ValueError(f"No image frames found in: {path}")
        h, w = frames[0].shape[:2]
        return {"fps": default_fps, "width": w, "height": h, "frame_count": len(frames)}
    raise FileNotFoundError(f"Not a file or directory: {path}")
