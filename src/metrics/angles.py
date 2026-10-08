"""Compute knee and hip angles over time from saved MediaPipe keypoints.

Usage: python src/metrics/angles.py outputs/<video>/mediapipe_keypoints.npz
Writes angles.csv and angles.png next to the input file.
Angles are measured in the image plane, so they are only meaningful for a
side view with the camera perpendicular to the movement.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SIDES = {
    "left":  dict(shoulder=11, hip=23, knee=25, ankle=27),
    "right": dict(shoulder=12, hip=24, knee=26, ankle=28),
}


def angle(a, b, c):
    """Angle at b, in degrees, between b->a and b->c. Inputs have shape (T, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def fill_nans(x):
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok])


def smooth(x, window=9, order=3):
    x = fill_nans(x)
    window = min(window, len(x) - (1 - len(x) % 2))  # must be odd and <= len(x)
    return savgol_filter(x, window, order)


def main(npz_path):
    d = np.load(npz_path)
    kp, fps = d["keypoints"], float(d["fps"])   # kp: (T, 33, 3) = x_px, y_px, visibility

    # Use the side of the body the camera sees best
    vis = {s: np.nanmean(kp[:, list(j.values()), 2]) for s, j in SIDES.items()}
    side = max(vis, key=vis.get)
    j = SIDES[side]
    print(f"Using {side} side (mean visibility: {vis[side]:.2f})")

    xy = lambda name: kp[:, j[name], :2]
    knee = smooth(angle(xy("hip"), xy("knee"), xy("ankle")))
    hip = smooth(angle(xy("shoulder"), xy("hip"), xy("knee")))
    t = np.arange(len(knee)) / fps

    out = Path(npz_path).parent
    pd.DataFrame({"time_s": t, "knee_deg": knee, "hip_deg": hip}).to_csv(out / "angles.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(t, knee, label="Knee angle")
    plt.plot(t, hip, label="Hip angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title(f"Joint angles ({side} side, image plane)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "angles.png", dpi=150)
    print(f"Knee range: {knee.min():.0f}-{knee.max():.0f} deg | Hip range: {hip.min():.0f}-{hip.max():.0f} deg")
    print(f"Saved angles.csv and angles.png to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    main(ap.parse_args().npz)