"""Estimate how noisy a joint-angle curve is, frame to frame, for one clip.

Usage:
  python src/metrics/jitter.py outputs/<video>/mediapipe_keypoints.npz --lift deadlift
  (--lift is squat, deadlift or bench, same as lift_reps.py)

Prints, for the three joints of that lift on the best-visible side:
  noise_deg      estimated frame-to-frame noise of the raw angle, in degrees
                 (robust estimate from the second difference of the angle)
  mean_vis       mean MediaPipe visibility of the three joints
  low_vis_pct    % of frames where any of the three joints has visibility < 0.5
Higher noise_deg = a less stable estimate. It is a rough comparison tool
between clips, not a measurement of true angle error.
"""
import argparse

import numpy as np

from lift_reps import LIFTS, LM, angle, fill_nans   # same folder as this script


def main(npz_path, lift):
    d = np.load(npz_path)
    kp = d["keypoints"]
    names = LIFTS[lift]["triple"]
    vis = {s: np.nanmean(kp[:, [LM[s][n] for n in names], 2]) for s in LM}
    side = max(vis, key=vis.get)
    pts = [kp[:, LM[side][n], :2] for n in names]
    raw = fill_nans(angle(*pts))                       # NOT smoothed
    d2 = raw[2:] - 2 * raw[1:-1] + raw[:-2]
    # white noise of std s gives second differences of std s*sqrt(6)
    noise = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    v = kp[:, [LM[side][n] for n in names], 2]
    low = float(np.mean(np.nanmin(v, axis=1) < 0.5) * 100)
    print(f"{npz_path}")
    print(f"  side={side}  noise_deg={noise:.2f}  mean_vis={vis[side]:.2f}  low_vis_pct={low:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--lift", required=True, choices=list(LIFTS))
    a = ap.parse_args()
    main(a.npz, a.lift)