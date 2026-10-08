"""Segment reps from one joint-angle curve, for squat, deadlift (hip hinge) or bench.

Usage:
  python src/metrics/lift_reps.py outputs/<video>/mediapipe_keypoints.npz --lift squat
  python src/metrics/lift_reps.py outputs/<video>/mediapipe_keypoints.npz --lift deadlift
  python src/metrics/lift_reps.py outputs/<video>/mediapipe_keypoints.npz --lift bench
Optional: --min-drop DEG  (smallest angle drop that counts as a rep)

Angle used (measured in the image plane, so only meaningful for a side view):
  squat     knee angle    hip - knee - ankle
  deadlift  hip angle     shoulder - hip - knee
  bench     elbow angle   shoulder - elbow - wrist
A rep is a dip in that angle. The bottom plateau (within --hold-tol degrees of the
minimum) is reported as hold_s; ascent_s starts when the plateau ends (liftoff_s). "Top zone" = within 10% of the full range of the
lockout/standing angle. descent_s is the time from leaving the top zone to the
bottom, ascent_s from the bottom back into it, so standing pauses are not
counted. complete=False means the clip started or ended before the angle was
back in the top zone, so that rep's timings are cut short.
Writes <lift>_reps.csv and <lift>_angle.png next to the input file.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

LM = {
    "left":  dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
    "right": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
}
LIFTS = {
    "squat":    dict(triple=("hip", "knee", "ankle"),        label="Knee angle",  min_drop=40),
    "deadlift": dict(triple=("shoulder", "hip", "knee"),     label="Hip angle",   min_drop=35),
    "bench":    dict(triple=("shoulder", "elbow", "wrist"),  label="Elbow angle", min_drop=40),
}


def angle(a, b, c):
    """Angle at b in degrees between b->a and b->c. Inputs have shape (T, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def fill_nans(x):
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok])


def smooth(x, window=9, order=3):
    x = fill_nans(x)
    window = min(window, len(x) - (1 - len(x) % 2))
    return savgol_filter(x, window, order)


def main(npz_path, lift, min_drop=None, hold_tol=8.0):
    cfg = LIFTS[lift]
    min_drop = cfg["min_drop"] if min_drop is None else min_drop
    d = np.load(npz_path)
    kp, fps = d["keypoints"], float(d["fps"])
    names = cfg["triple"]

    # Use the side of the body the camera sees best (for these three joints)
    vis = {s: np.nanmean(kp[:, [LM[s][n] for n in names], 2]) for s in LM}
    side = max(vis, key=vis.get)
    pts = {n: kp[:, LM[side][n], :2] for n in names}
    ang = smooth(angle(pts[names[0]], pts[names[1]], pts[names[2]]))
    t = np.arange(len(ang)) / fps
    n_missing = int(np.isnan(kp[:, LM[side][names[1]], 0]).sum())

    minima, _ = find_peaks(-ang, prominence=min_drop)

    # Top zone: near the lockout / standing angle of this clip
    top = float(np.percentile(ang, 95))
    bottom_ref = float(np.median(ang[minima])) if len(minima) else float(ang.min())
    thr = top - 0.10 * (top - bottom_ref)

    rows = []
    lows = [0] + list(minima[:-1])
    highs = list(minima[1:]) + [len(ang) - 1]
    for n, (m, lo, hi) in enumerate(zip(minima, lows, highs), 1):
        left = np.where(ang[lo:m] >= thr)[0]
        if len(left):
            start, ok_l = lo + int(left[-1]), True
        elif n == 1:
            start, ok_l = 0, False                      # clip began before the top zone
        else:
            start, ok_l = lo + int(np.argmax(ang[lo:m + 1])), True   # no full top between reps
        right = np.where(ang[m + 1:hi + 1] >= thr)[0]
        if len(right):
            end, ok_r = m + 1 + int(right[0]), True
        elif n == len(minima):
            end, ok_r = len(ang) - 1, False             # clip ended before the top zone
        else:
            end, ok_r = m + int(np.argmax(ang[m:hi + 1])), True
        # Bottom plateau: contiguous frames within hold_tol degrees of the minimum.
        # A pause at the bottom is reported as hold_s, not counted as ascent.
        p0 = p1 = int(m)
        while p0 > start and ang[p0 - 1] <= ang[m] + hold_tol:
            p0 -= 1
        while p1 < end and ang[p1 + 1] <= ang[m] + hold_tol:
            p1 += 1
        rows.append(dict(
            rep=n,
            bottom_time_s=round(m / fps, 2),
            min_deg=round(float(ang[m]), 1),
            descent_s=round((p0 - start) / fps, 2),
            hold_s=round((p1 - p0) / fps, 2),
            liftoff_s=round(p1 / fps, 2),
            ascent_s=round((end - p1) / fps, 2),
            complete=bool(ok_l and ok_r),
        ))

    out = Path(npz_path).parent
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{lift}_reps.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(t, ang, label=cfg["label"])
    plt.plot(t[minima], ang[minima], "ro", label="rep bottom")
    plt.axhline(thr, color="gray", ls=":", label="top-zone limit")
    for r in rows:
        plt.annotate(str(r["rep"]), (r["bottom_time_s"], r["min_deg"]),
                     textcoords="offset points", xytext=(0, -14), ha="center")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (degrees)")
    plt.title(f"{lift}: {cfg['label'].lower()} ({side} side, image plane)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{lift}_angle.png", dpi=150)

    print(f"Using {side} side (mean visibility {vis[side]:.2f}); "
          f"{n_missing} frame(s) with no detection on this joint.")
    print(f"Detected {len(df)} rep(s), {int(df.complete.sum()) if len(df) else 0} complete.")
    if len(df):
        print(df.to_string(index=False))
        print("complete=False: the clip started/ended before the angle was back at the top, so that rep's timing is cut short.")
    else:
        print("No reps found. Try a lower --min-drop.")
    print(f"Saved {lift}_reps.csv and {lift}_angle.png to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--lift", required=True, choices=list(LIFTS))
    ap.add_argument("--min-drop", type=float, default=None)
    ap.add_argument("--hold-tol", type=float, default=8.0, help="degrees from the minimum that count as the bottom plateau")
    a = ap.parse_args()
    main(a.npz, a.lift, a.min_drop, a.hold_tol)
