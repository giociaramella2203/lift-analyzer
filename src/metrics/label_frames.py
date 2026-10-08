"""Hand-label shoulder, hip and knee on selected frames, to measure pose-model error.

Usage:
  python src/metrics/label_frames.py <clip> [--lift deadlift] [--n 24] [--session A] [--scale 0.9]
                                     [--crop X0 Y0 X1 Y1] [--seed 0] [--video path] [--ignore 1]
Needs data/raw/<clip>.mp4 and outputs/<clip>/<lift>_reps.csv (to choose frames around the ascents).
Writes labels/<clip>_<session>.csv (pixel coordinates in the ORIGINAL frame) and resumes where you stopped.

For each frame, click in this order, on the leg nearest the camera (the side the models call "left"
when you face left in the image):
    1) shoulder   2) hip   3) knee
Keys:  u = undo last click   q = save and quit
       after 3 clicks:  ENTER or SPACE = accept (knee visible)   h = accept, knee HIDDEN (your best estimate)
       s = skip the frame (only when even an estimate is impossible)
Model predictions are NOT drawn, on purpose, so they cannot bias your clicks.

Tips
  * Hip and knee are joint CENTRES, not the skin outline. Be consistent across frames.
  * Behind the plate the knee is hidden: click your best estimate from the thigh above and the shin below,
    then press h (not ENTER) so the frame is marked as hidden-knee. Don't skip these frames: they are exactly
    the ones the comparison is about. These labels are less certain, so label the same frames again in a second session (--session B) on another
    day; eval_labels.py then reports how much your own clicks differ, which is the noise floor of the labels.
  * --crop zooms on a region (e.g. --crop 0 300 576 1024) so you can click more precisely.
Frames: about 70% from the ascents (plate crossing the knee), 30% around the bottom of the reps.
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

JOINTS = ["shoulder", "hip", "knee"]
FIELDS = ["clip", "session", "frame", "time_s", "phase", "status", "knee_hidden",
          "shoulder_x", "shoulder_y", "hip_x", "hip_y", "knee_x", "knee_y"]


def pick_frames(reps_csv, fps, total, n, seed, min_gap=8, ignore=()):
    """Return a sorted list of (frame, phase). phase is 'ascent' or 'bottom'."""
    rng = random.Random(seed)
    asc, low = [], []
    if Path(reps_csv).exists():
        for r in pd.read_csv(reps_csv).itertuples():
            if r.rep in ignore:
                continue
            a, b = int(r.liftoff_s * fps), int((r.liftoff_s + r.ascent_s) * fps)
            asc += list(range(a, min(b, total - 1) + 1))
            c = int(r.bottom_time_s * fps)
            low += list(range(max(0, c - 8), min(c + 8, total - 1) + 1))
    if not asc:
        asc = list(range(total))
    chosen = []

    def take(pool, k, phase):
        pool = pool[:]
        rng.shuffle(pool)
        out = []
        for f in pool:
            if all(abs(f - g) >= min_gap for g, _ in chosen + out):
                out.append((f, phase))
            if len(out) == k:
                break
        return out

    n_asc = int(round(n * 0.7))
    chosen += take(asc, n_asc, "ascent")
    chosen += take(low, n - len(chosen), "bottom")
    return sorted(chosen)


def main(clip, lift, n, session, scale, crop, seed, video, ignore=()):
    video = Path(video) if video else Path("data/raw") / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = crop if crop else (0, 0, W, H)

    out_path = Path("labels") / f"{clip}_{session}.csv"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["frame"].tolist())
    else:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    todo = [(f, p) for f, p in pick_frames(Path("outputs") / clip / f"{lift}_reps.csv", fps, total, n, seed, ignore=set(ignore)) if f not in done]
    print(f"{len(todo)} frame(s) to label ({len(done)} already done). Session {session}.")
    win = f"label {clip} [{session}]"
    cv2.namedWindow(win)
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 3:
            pts.append((x / scale + x0, y / scale + y0))

    cv2.setMouseCallback(win, on_mouse)
    for k, (frame_idx, phase) in enumerate(todo, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            continue
        pts.clear()
        while True:
            view = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale)
            for i, (px, py) in enumerate(pts):
                c = (int((px - x0) * scale), int((py - y0) * scale))
                cv2.circle(view, c, 5, (0, 255, 255), -1)
                cv2.putText(view, JOINTS[i], (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            nxt = JOINTS[len(pts)] if len(pts) < 3 else "ENTER = knee visible, h = knee hidden"
            msg = f"{k}/{len(todo)}  frame {frame_idx}  t={frame_idx / fps:.1f}s  click: {nxt}   (u undo, h hidden, s skip, q quit)"
            cv2.rectangle(view, (0, 0), (view.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(view, msg, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("u") and pts:
                pts.pop()
            elif key == ord("q"):
                cv2.destroyAllWindows()
                print(f"Saved to {out_path}")
                return
            elif key in (13, 32, ord("h")) and len(pts) == 3:
                row = dict(clip=clip, session=session, frame=frame_idx, time_s=round(frame_idx / fps, 3), phase=phase,
                           status="ok", knee_hidden=int(key == ord("h")))
                for j, (px, py) in zip(JOINTS, pts):
                    row[f"{j}_x"], row[f"{j}_y"] = round(px, 1), round(py, 1)
                break
            elif key == ord("s"):
                row = dict(clip=clip, session=session, frame=frame_idx, time_s=round(frame_idx / fps, 3), phase=phase, status="skip", knee_hidden="")
                break
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
    cv2.destroyAllWindows()
    print(f"Done. Saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", default="deadlift")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--session", default="A")
    ap.add_argument("--scale", type=float, default=0.9)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--ignore", type=int, nargs="*", default=[], help="rep numbers to leave out (e.g. 1 for a walk-in)")
    a = ap.parse_args()
    main(a.clip, a.lift, a.n, a.session, a.scale, a.crop, a.seed, a.video, a.ignore)
