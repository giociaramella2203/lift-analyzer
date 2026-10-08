"""Extract 33 body landmarks per frame with MediaPipe PoseLandmarker.

Saves keypoints to outputs/<video_name>/mediapipe_keypoints.npz and an
annotated video to outputs/<video_name>/mediapipe_annotated.mp4.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Skeleton edges between the 33 MediaPipe landmark indices
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def extract(video_path, model_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_dir / "mediapipe_annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
    )

    kp_img, kp_world = [], []   # (x_px, y_px, visibility) and (x, y, z) in metres
    n_missed = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(i * 1000 / fps))

            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                wl = result.pose_world_landmarks[0]
                pts = np.array([[p.x * w, p.y * h, p.visibility] for p in lm])
                kp_img.append(pts)
                kp_world.append(np.array([[p.x, p.y, p.z] for p in wl]))
                for a, b in CONNECTIONS:
                    cv2.line(frame, tuple(pts[a, :2].astype(int)),
                             tuple(pts[b, :2].astype(int)), (0, 255, 0), 2)
                for x, y, _ in pts:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            else:
                n_missed += 1
                kp_img.append(np.full((33, 3), np.nan))
                kp_world.append(np.full((33, 3), np.nan))

            writer.write(frame)
            i += 1

    cap.release()
    writer.release()
    np.savez(out_dir / "mediapipe_keypoints.npz",
             keypoints=np.stack(kp_img), world=np.stack(kp_world),
             fps=fps, width=w, height=h)
    print(f"{i} frames at {fps:.1f} fps, {w}x{h}. No person found in {n_missed} frames.")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="models/pose_landmarker_heavy.task")
    args = ap.parse_args()
    name = Path(args.video).stem
    extract(args.video, args.model, Path("outputs") / name)