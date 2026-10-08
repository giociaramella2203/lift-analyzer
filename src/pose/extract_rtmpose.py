"""Run RTMPose (via rtmlib, ONNX, CPU) on a video and save keypoints in the SAME
format as extract_mediapipe.py, so lift_reps.py, jitter.py and hud_video.py work unchanged.

Usage:
  python src/pose/extract_rtmpose.py data/raw/deadlift_set2.mp4 [--mode balanced]
Writes outputs/<clip>_rtm/mediapipe_keypoints.npz and mediapipe_annotated.mp4
(the file names keep the "mediapipe_" prefix on purpose, so the other scripts find them).

Then run the usual scripts on the folder <clip>_rtm, e.g.
  python src/metrics/lift_reps.py outputs/deadlift_set2_rtm/mediapipe_keypoints.npz --lift deadlift
  python src/metrics/compare_models.py deadlift_set2 --lift deadlift

Notes
  * RTMPose gives 17 COCO keypoints. Only nose, shoulders, elbows, wrists, hips, knees
    and ankles are copied (into the MediaPipe index slots those scripts expect);
    every other slot stays NaN.
  * The "visibility" column holds RTMPose's keypoint score. It is NOT on the same scale
    as MediaPipe visibility, so do not compare confidence numbers across the two models.
  * The first run downloads the ONNX weights (needs internet). mode: lightweight (fast),
    balanced (default), performance (slowest, most accurate).
  * If several people are in the frame, the largest, most confident one is used.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body

# COCO-17 index -> MediaPipe-33 index
COCO_TO_MP = {0: 0, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]


def pick_person(kp, sc, thr=0.3):
    """Index of the largest, most confident detected person, or None."""
    best, best_val = None, -1.0
    for i in range(len(kp)):
        ok = sc[i] > thr
        if ok.sum() < 5:
            continue
        pts = kp[i][ok]
        val = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]) * sc[i][ok].mean())
        if val > best_val:
            best, best_val = i, val
    return best


def main(video, mode):
    clip = Path(video).stem
    out = Path("outputs") / f"{clip}_rtm"
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out / "mediapipe_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = Body(mode=mode, backend="onnxruntime", device="cpu")
    rows, no_person, i = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kp, sc = model(frame)
        k = pick_person(kp, sc) if len(kp) else None
        row = np.full((33, 3), np.nan)
        row[:, 2] = 0.0
        if k is None:
            no_person += 1
        else:
            for c, m in COCO_TO_MP.items():
                row[m, 0], row[m, 1], row[m, 2] = kp[k][c][0], kp[k][c][1], sc[k][c]
            for a, b in SKELETON:
                cv2.line(frame, tuple(int(v) for v in kp[k][a]), tuple(int(v) for v in kp[k][b]), (0, 255, 0), 2)
            for c in COCO_TO_MP:
                cv2.circle(frame, tuple(int(v) for v in kp[k][c]), 3, (0, 0, 255), -1)
        rows.append(row)
        writer.write(frame)
        i += 1
        if i % 100 == 0:
            print(f"  {i}/{n} frames", flush=True)
    cap.release()
    writer.release()

    keypoints = np.stack(rows)
    np.savez(out / "mediapipe_keypoints.npz", keypoints=keypoints, world=np.zeros_like(keypoints),
             fps=fps, width=w, height=h)
    print(f"{len(keypoints)} frames, {fps:.1f} fps, {w}x{h}; no person in {no_person} frame(s).")
    print(f"Saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    a = ap.parse_args()
    main(a.video, a.mode)
