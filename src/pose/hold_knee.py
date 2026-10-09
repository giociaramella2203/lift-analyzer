"""Control for the point tracker: hold the knee still. No video, no tracking, no model beyond the pose model.

For every rep, the knee position the pose model gives at the bottom of the rep (the same frame the tracker is anchored
on) is simply copied to every frame of that rep. If the tracker does not beat this, it adds nothing beyond a frozen
position (the knee moves little in the image during a deadlift).

Usage:  python src/pose/hold_knee.py <clip> [--lift deadlift] [--init-folder <clip>_rtm] [--side left] [--ignore 1]
Needs data/raw/<clip>.mp4 (for the frame rate), outputs/<clip>/<lift>_reps.csv and outputs/<init folder>/mediapipe_keypoints.npz.
Writes outputs/<clip>_hold/mediapipe_keypoints.npz, then score it like any model:
  python src/metrics/eval_labels.py <clip> --models <clip> <clip>_rtm <clip>_track2 <clip>_hold --debias
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_knee import rep_segments, rep_bottoms, LM  # noqa: E402


def main(clip, lift, init_folder, side, ignore):
    fps = cv2.VideoCapture(str(Path("data/raw") / f"{clip}.mp4")).get(cv2.CAP_PROP_FPS) or 30.0
    reps = Path("outputs") / clip / f"{lift}_reps.csv"
    kp = np.load(Path("outputs") / init_folder / "mediapipe_keypoints.npz")["keypoints"].astype(float)
    out = kp.copy()
    k = LM[side]["knee"]
    bottoms = rep_bottoms(reps, fps)
    for rep, s, e in rep_segments(reps, fps, len(kp), ignore):
        q = int(np.clip(bottoms.get(rep, s), s, e))
        if np.all(np.isfinite(kp[q, k, :2])):
            out[s:e + 1, k, :2] = kp[q, k, :2]
            print(f"  rep {rep}: frames {s}-{e}, knee held at frame {q}: {kp[q, k, :2].round()}")
    folder = Path("outputs") / f"{clip}_hold"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "mediapipe_keypoints.npz", keypoints=out)
    print(f"Saved {folder / 'mediapipe_keypoints.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", default="deadlift")
    ap.add_argument("--init-folder", default=None)
    ap.add_argument("--side", default="left")
    ap.add_argument("--ignore", nargs="*", type=int, default=[])
    a = ap.parse_args()
    main(a.clip, a.lift, a.init_folder or f"{a.clip}_rtm", a.side, set(a.ignore))
