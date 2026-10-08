"""Segment reps and estimate squat depth from saved MediaPipe keypoints.

Usage: python src/metrics/reps.py outputs/<video>/mediapipe_keypoints.npz
Optional: --min-drop 40 --threshold 0.0 --tol 0.15
Writes reps.csv next to the input file and prints a summary.

Depth is reported as a continuous margin: how far the hip landmark sits below
the knee landmark at the bottom of each rep, as a fraction of thigh length
(positive = hip below knee, negative = hip above knee).

depth_call turns the margin into three outcomes:
  deep        margin > threshold + tol
  shallow     margin < threshold - tol
  borderline  in between
The pose model gives joint centres, not the hip crease and top of the knee
used in competition, so the threshold should be calibrated on lifts where the
true outcome is known. The call is an estimate, not a ruling.
Only meaningful for a side view with the camera at about hip height.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

SIDES = {
    "left":  dict(shoulder=11, hip=23, knee=25, ankle=27),
    "right": dict(shoulder=12, hip=24, knee=26, ankle=28),
}


def fill_nans(x):
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok])


def smooth(x, window=9, order=3):
    x = fill_nans(x)
    window = min(window, len(x) - (1 - len(x) % 2))
    return savgol_filter(x, window, order)


def knee_angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main(npz_path, min_drop=40, threshold=0.0, tol=0.15):
    d = np.load(npz_path)
    kp, fps = d["keypoints"], float(d["fps"])

    vis = {s: np.nanmean(kp[:, list(j.values()), 2]) for s, j in SIDES.items()}
    side = max(vis, key=vis.get)
    j = SIDES[side]
    hip, knee, ankle = (kp[:, j[n], :2] for n in ("hip", "knee", "ankle"))

    ang = smooth(knee_angle(hip, knee, ankle))
    hip_y = smooth(hip[:, 1])
    knee_y = smooth(knee[:, 1])
    thigh_len = np.nanmedian(np.linalg.norm(hip - knee, axis=1))

    # A rep = a dip in the knee angle. Standing is the top of the curve, so a
    # rep is a minimum of the angle that drops at least `min_drop` degrees.
    minima, _ = find_peaks(-ang, prominence=min_drop)
    maxima, _ = find_peaks(ang, prominence=min_drop)

    rows = []
    for n, m in enumerate(minima, 1):
        before = maxima[maxima < m]
        after = maxima[maxima > m]
        start = before[-1] if len(before) else 0
        end = after[0] if len(after) else len(ang) - 1
        # depth margin: positive = hip below knee. Normalised by thigh length
        margin = (hip_y[m] - knee_y[m]) / thigh_len
        rows.append(dict(
            rep=n,
            min_knee_deg=round(float(ang[m]), 1),
            descent_s=round((m - start) / fps, 2),
            ascent_s=round((end - m) / fps, 2),
            depth_margin=round(float(margin), 2),
            depth_call=("deep" if margin > threshold + tol
                        else "shallow" if margin < threshold - tol
                        else "borderline"),
            bottom_time_s=round(m / fps, 2),
        ))

    out = Path(npz_path).parent
    df = pd.DataFrame(rows)
    df.to_csv(out / "reps.csv", index=False)
    print(f"Using {side} side. Detected {len(df)} rep(s).")
    if len(df):
        print(df.to_string(index=False))
        print("depth_margin: fraction of thigh length the hip is below the knee (negative = above).")
        print(f"depth_call uses threshold={threshold} and tol={tol}.")
    else:
        print("No reps found. Try a lower --min-drop.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--min-drop", type=float, default=40)
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="depth_margin that counts as depth (negative = hip may sit above knee centre)")
    ap.add_argument("--tol", type=float, default=0.15,
                    help="half-width of the 'borderline' band around the threshold")
    a = ap.parse_args()
    main(a.npz, a.min_drop, a.threshold, a.tol)