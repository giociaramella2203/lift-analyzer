"""Clean keypoints for one lift: remove physically impossible frames, then fill the gaps.

Usage:
  python src/metrics/clean_keypoints.py <folder_name> --lift deadlift [--tol 0.20] [--max-gap 20]
  e.g.  python src/metrics/clean_keypoints.py deadlift_set2 --lift deadlift
        python src/metrics/clean_keypoints.py deadlift_set2_rtm --lift deadlift
Reads  outputs/<folder>/mediapipe_keypoints.npz
Writes outputs/<folder>_clean/ (cleaned keypoints, a copy of the annotated video, clean_keypoints.png)

What it does (only for the three joints of the chosen lift, on the best-visible side):
  1. Bone-length check. The two bones of the lift (deadlift: torso and thigh; squat: thigh and shank;
     bench: upper arm and forearm) should keep a roughly constant length. Each is compared with its
     own rolling median (about 3 s). A frame is bad if a bone is off by more than --tol (default 20%).
       only the second bone off  -> its far joint is blamed (deadlift: the knee)
       only the first bone off   -> its far joint is blamed (deadlift: the shoulder)
       both off                  -> the shared middle joint is blamed (deadlift: the hip)
  2. Single-frame spikes are removed with a Hampel filter (window 9, 3.5 MADs).
  3. Gaps of up to --max-gap frames are filled by linear interpolation; longer gaps stay empty.
The other landmarks are left untouched. Prints how much was changed and the angle noise before/after.

IMPORTANT: this removes impossible geometry and smooths noise. It does NOT prove the remaining
positions are correct; only hand-labelled frames can do that. Side views only: the bone-length
assumption is weaker when the limb points toward or away from the camera.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lift_reps import LIFTS, LM, angle, fill_nans


def rolling_median(x, win=91):
    return pd.Series(x).rolling(win, center=True, min_periods=15).median().to_numpy()


def hampel(x, win=9, k=3.5):
    """Mark single-frame outliers of a 1-D series (NaN-safe). Returns a boolean mask."""
    s = pd.Series(x)
    med = s.rolling(win, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=3).median() * 1.4826
    return ((s - med).abs() > k * mad.clip(lower=1.0)).fillna(False).to_numpy()


def interp_short(x, max_gap):
    """Linear interpolation of NaN runs no longer than max_gap; longer runs stay NaN."""
    x = x.copy()
    n, i = len(x), 0
    while i < n:
        if np.isnan(x[i]):
            j = i
            while j + 1 < n and np.isnan(x[j + 1]):
                j += 1
            if (j - i + 1) <= max_gap and i > 0 and j < n - 1 and not np.isnan(x[i - 1]) and not np.isnan(x[j + 1]):
                x[i:j + 1] = np.interp(np.arange(i, j + 1), [i - 1, j + 1], [x[i - 1], x[j + 1]])
            i = j + 1
        else:
            i += 1
    return x


def noise_deg(pts):
    raw = fill_nans(angle(*pts))
    d2 = raw[2:] - 2 * raw[1:-1] + raw[:-2]
    return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)


def main(folder, lift, tol, max_gap):
    base = Path("outputs")
    src = base / folder / "mediapipe_keypoints.npz"
    d = np.load(src)
    kp = d["keypoints"].copy()
    fps = float(d["fps"])
    names = LIFTS[lift]["triple"]                      # (a, b, c): bones a-b and b-c
    vis = {s: np.nanmean(kp[:, [LM[s][n] for n in names], 2]) for s in LM}
    side = max(vis, key=vis.get)
    idx = [LM[side][n] for n in names]
    P = [kp[:, i, :2].copy() for i in idx]             # raw copies
    before = noise_deg(P)

    lens = [np.linalg.norm(P[0] - P[1], axis=1), np.linalg.norm(P[1] - P[2], axis=1)]
    bad = []
    for L in lens:
        ref = rolling_median(L)
        bad.append(np.abs(L / ref - 1) > tol)
    bad_ab, bad_bc = bad
    blame = [np.zeros(len(kp), bool) for _ in range(3)]
    blame[2] = bad_bc & ~bad_ab
    blame[0] = bad_ab & ~bad_bc
    blame[1] = bad_ab & bad_bc

    cleaned = []
    for j in range(3):
        xy = P[j].copy()
        xy[blame[j]] = np.nan
        for c in range(2):
            spike = hampel(xy[:, c])
            xy[spike, c] = np.nan
            xy[:, c] = interp_short(xy[:, c], max_gap)
        both = np.isnan(xy).any(axis=1)
        xy[both] = np.nan
        cleaned.append(xy)

    for j, i in enumerate(idx):
        kp[:, i, :2] = cleaned[j]
    after = noise_deg(cleaned)

    out = base / f"{folder}_clean"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mediapipe_keypoints.npz", keypoints=kp, world=d["world"], fps=fps,
             width=d["width"], height=d["height"])
    ann = base / folder / "mediapipe_annotated.mp4"
    if ann.exists() and not (out / "mediapipe_annotated.mp4").exists():
        shutil.copy(ann, out / "mediapipe_annotated.mp4")

    T = len(kp)
    print(f"{folder} / {lift}  side={side}  ({T} frames)")
    for j, n in enumerate(names):
        changed = np.isnan(cleaned[j]).any(axis=1) | (np.linalg.norm(cleaned[j] - P[j], axis=1) > 0.5)
        print(f"  {n:9s}: {blame[j].sum():4d} frames blamed by bone length; "
              f"{changed.sum():4d} changed in total ({changed.mean() * 100:.1f}%); "
              f"{np.isnan(cleaned[j]).any(axis=1).sum()} left empty")
    print(f"  angle noise: {before:.2f} deg -> {after:.2f} deg")
    cv = lambda L: np.nanstd(L) / np.nanmedian(L) * 100
    new_lens = [np.linalg.norm(cleaned[0] - cleaned[1], axis=1), np.linalg.norm(cleaned[1] - cleaned[2], axis=1)]
    print(f"  bone-length spread (std/median): {cv(lens[0]):.1f}% -> {cv(new_lens[0]):.1f}% ({names[0]}-{names[1]}),  "
          f"{cv(lens[1]):.1f}% -> {cv(new_lens[1]):.1f}% ({names[1]}-{names[2]})")
    print("  The remaining positions are NOT verified: validate against hand-labelled frames.")

    t = np.arange(T) / fps
    j = 2   # third joint (knee for deadlift)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, lab in enumerate(("x (px)", "y (px, down = larger)")):
        ax[k].plot(t, P[j][:, k], color="tab:red", alpha=0.6, lw=1, label=f"{names[j]} before")
        ax[k].plot(t, cleaned[j][:, k], color="tab:blue", lw=1.2, label=f"{names[j]} after")
        ax[k].scatter(t[blame[j]], P[j][blame[j], k], s=8, color="k", zorder=3, label="blamed by bone length")
        ax[k].set_ylabel(lab)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Time (s)")
    ax[0].legend(ncol=3, loc="upper right")
    fig.suptitle(f"{folder}: {names[j]} before/after cleaning")
    fig.tight_layout()
    fig.savefig(out / "clean_keypoints.png", dpi=140)
    print(f"Saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--lift", required=True, choices=list(LIFTS))
    ap.add_argument("--tol", type=float, default=0.20, help="allowed bone-length deviation (fraction)")
    ap.add_argument("--max-gap", type=int, default=20, help="longest gap (frames) to fill by interpolation")
    a = ap.parse_args()
    main(a.folder, a.lift, a.tol, a.max_gap)
