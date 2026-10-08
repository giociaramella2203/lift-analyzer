"""Plot the LEFT and RIGHT knee paths for MediaPipe and RTMPose on the same clip.

Usage:
  python src/metrics/knee_paths.py <clip_name> [--lift deadlift]
Needs outputs/<clip>/mediapipe_keypoints.npz and outputs/<clip>_rtm/mediapipe_keypoints.npz.
If outputs/<clip>/<lift>_reps.csv exists, the ascent of each rep (lift-off to top) is shaded.

Why: when a plate hides a leg, a model may (a) confuse the near and far leg, or (b) guess the
hidden joint. A near/far swap shows up as the LEFT trace jumping onto the RIGHT trace.
Prints, per model, how many frames look like a possible swap jump: the left knee is closer to where the
right knee was one frame earlier than to where the left knee was, while the two legs are at least
15 px apart. It counts the jump itself, not the frames spent on the wrong leg. This is a heuristic
flag, not proof: look at the plot and the video.
Writes outputs/<clip>/knee_paths.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LK, RK = 25, 26   # MediaPipe landmark indices for left / right knee


def load(path):
    d = np.load(path)
    kp = d["keypoints"]
    return kp[:, LK, :2], kp[:, RK, :2], float(d["fps"])


def swaps(L, R, min_sep=15.0):
    flags = np.zeros(len(L), bool)
    for t in range(1, len(L)):
        if np.isnan([L[t], L[t - 1], R[t - 1]]).any():
            continue
        sep = np.linalg.norm(L[t - 1] - R[t - 1])
        if sep >= min_sep and np.linalg.norm(L[t] - R[t - 1]) < np.linalg.norm(L[t] - L[t - 1]):
            flags[t] = True
    return flags


def main(clip, lift):
    base = Path("outputs")
    models = {"MediaPipe": base / clip / "mediapipe_keypoints.npz", "RTMPose": base / f"{clip}_rtm" / "mediapipe_keypoints.npz"}
    windows = []
    csv = base / clip / f"{lift}_reps.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        windows = [(r.liftoff_s, r.liftoff_s + r.ascent_s) for r in df.itertuples()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for row, (name, path) in enumerate(models.items()):
        L, R, fps = load(path)
        t = np.arange(len(L)) / fps
        sw = swaps(L, R)
        in_asc = np.zeros(len(L), bool)
        for a, b in windows:
            in_asc |= (t >= a) & (t <= b)
        pct = f"{sw.sum()} possible swap jumps" + (f", {int((sw & in_asc).sum())} of them during ascents" if windows else "")
        print(f"{name:9s}: {pct}  (ascent frames: {int(in_asc.sum())} of {len(L)})")
        for col, (idx, lab) in enumerate(((0, "knee x (px)"), (1, "knee y (px, down = larger)"))):
            ax = axes[row, col]
            for a, b in windows:
                ax.axvspan(a, b, color="gray", alpha=0.15)
            ax.plot(t, L[:, idx], label="left knee", lw=1.2)
            ax.plot(t, R[:, idx], label="right knee", lw=1.2)
            ax.set_ylabel(lab)
            ax.set_title(f"{name}: {lab.split(' (')[0]}")
            if idx == 1:
                ax.invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper center", ncol=2)
    fig.suptitle(f"{clip}: left vs right knee, gray = ascents")
    fig.tight_layout()
    out = base / clip / "knee_paths.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", default="deadlift")
    a = ap.parse_args()
    main(a.clip, a.lift)
