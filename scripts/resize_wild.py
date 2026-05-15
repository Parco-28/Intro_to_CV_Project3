"""Resize wild video frames to fit within 480p for memory-safe processing."""
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.video_io import extract_frames, get_video_info
from utils.mask_utils import imwrite_unicode

video = "data/wild_video/wild_video.mp4"
out_dir = "data/wild_video/frames_480p"
os.makedirs(out_dir, exist_ok=True)

frames = extract_frames(video)
info = get_video_info(video)
h, w = frames[0].shape[:2]
max_dim = 480
scale = max_dim / max(h, w)
nh, nw = int(h * scale), int(w * scale)
print(f"Original: {w}x{h}, Resized: {nw}x{nh}, {len(frames)} frames @ {info['fps']:.1f} FPS")

for i, f in enumerate(frames):
    resized = cv2.resize(f, (nw, nh))
    imwrite_unicode(os.path.join(out_dir, f"{i:05d}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"Saved {len(frames)} resized frames to {out_dir}")
