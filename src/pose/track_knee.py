"""Does temporal memory help when a joint is hidden? Follow the knee with a point tracker (CoTracker3).

For every rep, the knee position given by a pretrained pose model on a frame where the knee is visible (the bottom of the
rep, --anchor bottom) is handed to CoTracker3, which follows that point forward and backward through the rest of the
rep, including the frames where the plate covers it. The tracked path replaces the model's knee on those frames. Nothing is trained.

The output has the same format as the pose models, so the existing scoring works unchanged:
  python src/metrics/eval_labels.py deadlift_set2 --models deadlift_set2 deadlift_set2_rtm deadlift_set2_track --debias --detail

Usage (needs a GPU for reasonable speed: run it on Google Colab):
  python track_knee.py --video deadlift_set2.mp4 --init mediapipe_keypoints.npz --reps deadlift_reps.csv --out deadlift_set2_track
    --init  keypoints from the pose model that provides the starting point (RTMPose folder recommended)
    --reps  the <lift>_reps.csv written by lift_reps.py
    --out   folder to create; it gets mediapipe_keypoints.npz (copy it to outputs/<clip>_track/ on your PC)
  options: --side left  --joints knee (default; "shoulder hip knee" tracks all three)  --ignore 1 (skip rep numbers)
           --max-side 768 (downscale before tracking, for speed)  --device cuda|cpu

Protocol history (settings are fixed before the second set of videos is filmed; do not tune on new footage):
  v1 (failed on deadlift_set2): anchor = first frame of the rep. In this camera view the plate already covers the
     knee at lockout, so the tracker was given a point on the plate and followed the plate to the floor.
  v2: anchor = the bottom of the rep (hip angle minimum), where the knee is above the plate; the tracker runs
     backward and forward from there through the stretch where the plate rises past the knee.
  Both: init model RTMPose, joints knee only, side left, CoTracker3 offline.
Caveat: the tracker inherits the starting model's convention offset (where it puts the joint centre); the
--debias table in eval_labels removes a constant offset, so compare the debiased numbers.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LM = {"left": {"shoulder": 11, "hip": 23, "knee": 25}, "right": {"shoulder": 12, "hip": 24, "knee": 26}}


def rep_segments(reps_csv, fps, total, ignore=()):
    """(rep, start_frame, end_frame) per rep: from the first frame of the rep to the end of the ascent."""
    df = pd.read_csv(reps_csv)
    segs = []
    for _, r in df.iterrows():
        if int(r["rep"]) in ignore:
            continue
        start = int(round((r["liftoff_s"] - r["hold_s"] - r["descent_s"]) * fps))
        end = int(round((r["liftoff_s"] + r["ascent_s"]) * fps))
        start, end = max(start, 0), min(end, total - 1)
        if end - start >= 5:
            segs.append((int(r["rep"]), start, end))
    return segs


def rep_bottoms(reps_csv, fps):
    """rep number -> frame of the lowest point of the rep (hip angle minimum)."""
    df = pd.read_csv(reps_csv)
    return {int(r["rep"]): int(round(r["bottom_time_s"] * fps)) for _, r in df.iterrows()}


def read_frames(cap, start, end, scale):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end + 1):
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def load_model(device):
    import torch
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    return model.to(device)


def run_model(model, frames, points, device, qt=0):
    """frames: list of RGB uint8 images; points: (N,2) x,y on frame qt. Tracks forward and backward from qt.
    Returns tracks (T,N,2) and visibility (T,N)."""
    import torch
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    q = torch.tensor([[[float(qt), x, y] for x, y in points]], dtype=torch.float32, device=device)
    with torch.no_grad():
        tracks, vis = model(video, queries=q, backward_tracking=True)
    return tracks[0].cpu().numpy(), vis[0].cpu().numpy().astype(float)


def main(video, init, reps, out, side, joints, ignore, max_side, device, model=None, anchor="bottom"):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kp = np.load(init)["keypoints"].astype(float)
    total = len(kp)
    scale = min(1.0, max_side / max(W, H))
    segs = rep_segments(reps, fps, total, ignore)
    bottoms = rep_bottoms(reps, fps)
    print(f"{video}: {W}x{H}, {fps:.1f} fps, {total} frames; {len(segs)} rep segment(s); tracking {', '.join(joints)} ({side}); scale {scale:.2f}")
    if model is None:
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("  No GPU found: this will be slow. On Colab choose Runtime > Change runtime type > GPU.")
        model = load_model(device)
    out_kp = kp.copy()
    idx = [LM[side][j] for j in joints]
    for rep, start, end in segs:
        q_abs = start if anchor == "start" else int(np.clip(bottoms.get(rep, start), start, end))
        pts = kp[q_abs, idx, :2]
        if not np.all(np.isfinite(pts)):
            print(f"  rep {rep}: no starting point (model missing at frame {q_abs}), skipped")
            continue
        frames = read_frames(cap, start, end, scale)
        n = len(frames)
        tracks, vis = run_model(model, frames, pts * scale, device, qt=q_abs - start)
        tracks = tracks / scale
        out_kp[start:start + n, idx, :2] = np.transpose(tracks, (0, 1, 2))
        out_kp[start:start + n, idx, 2] = vis
        move = np.nanmax(np.linalg.norm(tracks - tracks[q_abs - start], axis=2))
        print(f"  rep {rep}: frames {start}-{start + n - 1}, anchored at frame {q_abs}, max move from anchor {move:.0f} px, visible {np.mean(vis > 0.5) * 100:.0f}% of frames")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mediapipe_keypoints.npz", keypoints=out_kp)
    print(f"Saved {out / 'mediapipe_keypoints.npz'}  (copy it to outputs/<clip>_track/ on your PC)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--joints", nargs="+", default=["knee"], choices=["shoulder", "hip", "knee"])
    ap.add_argument("--ignore", nargs="*", type=int, default=[])
    ap.add_argument("--anchor", default="bottom", choices=["bottom", "start"], help="frame where the tracker is given the model's knee")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    main(a.video, a.init, a.reps, a.out, a.side, a.joints, set(a.ignore), a.max_side, a.device, anchor=a.anchor)
