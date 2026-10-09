"""Oracle-masking experiment: can a lifter (or plain interpolation) repair the hip angle when the knee is marked as missing?

Usage (from the repository root):
  python src/metrics/eval_masking.py <clip> <lift3d_<clip>_masked.npz> [--side left]
The masked file comes from the last cell of notebooks/lift3d_motionbert.ipynb. Needs outputs/<clip>_rtm/h36m_2d.npz
(with hidden_frames, from export_h36m.py) and labels/<clip>_*.csv.

The knee is marked missing in a window of +-K frames around each frame where I marked it hidden. THIS USES MY LABELS TO SAY WHERE
THE KNEE IS LOST, so it is an upper bound on what a lifter could do, not a usable method.
Variants: base (no masking), conf0 (knee confidence set to 0), zero (knee x,y and confidence set to 0),
interp (knee x,y replaced by linear interpolation from frames outside the window), and "2d_interp" (the same interpolation
without any lifter: the 2D hip angle from RTMPose with the interpolated knee).
Error = hip angle against my 2D labels, degrees, mean bias removed (leave-one-out), as in eval_lift3d.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle                     # noqa: E402
from eval_labels import load_labels, J          # noqa: E402
from world3d import debiased_abs_err            # noqa: E402

H36M = {"left": dict(shoulder=11, hip=4, knee=5), "right": dict(shoulder=14, hip=1, knee=2)}


def window(T, hidden, K):
    m = np.zeros(T, bool)
    for f in hidden:
        m[max(0, f - K):min(T, f + K + 1)] = True
    return m


def main(clip, masked_path, side):
    idx = H36M[side]
    z = np.load(Path("outputs") / f"{clip}_rtm" / "h36m_2d.npz")
    d2, hid = z["kp2d"], z["hidden_frames"]
    out = np.load(masked_path)
    T = len(d2)
    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    gt = gt[gt.frame < T]
    fr = gt.frame.to_numpy()
    hidden = gt.knee_hidden.to_numpy() == 1
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ag = angle(S["shoulder"], S["hip"], S["knee"])
    A = lambda arr, dims=3: angle(arr[:, idx["shoulder"], :dims], arr[:, idx["hip"], :dims], arr[:, idx["knee"], :dims])
    print(f"{clip}: {len(fr)} labelled frames ({int(hidden.sum())} knee-hidden); hidden frames used for masking: {hid.tolist()}")

    rows = {"RTMPose 2D, no masking": debiased_abs_err(A(d2[fr], 2), ag)}
    for key in out.files:
        rows[f"lifted 3D, {key}"] = debiased_abs_err(A(out[key][fr]), ag)
    for K in sorted({int(k.split("_K")[1]) for k in out.files if "_K" in k}):
        m = window(T, hid, K)
        y = d2.copy()
        t = np.arange(T)
        for c in (0, 1):
            y[m, idx["knee"], c] = np.interp(t[m], t[~m], d2[~m, idx["knee"], c])
        rows[f"2D, knee interpolated, K{K}"] = debiased_abs_err(A(y[fr], 2), ag)

    med = lambda e: f"{np.median(e):5.2f}" if len(e) else "  n/a"
    print(f"\n{'':36s}{'all':>7s}{'hidden':>8s}{'visible':>9s}   (hip-angle error, deg, median)")
    for name, e in rows.items():
        print(f"{name:36s}{med(e):>7s}{med(e[hidden]):>8s}{med(e[~hidden]):>9s}")

    base = rows["lifted 3D, base"]
    rng = np.random.default_rng(0)
    print("\nMedian of (variant minus lifted base) over ALL labelled frames, 95% bootstrap interval; and over hidden frames only")
    for name, e in rows.items():
        if name in ("lifted 3D, base", "RTMPose 2D, no masking"):
            continue
        diff = e - base
        b = [np.median(rng.choice(diff, len(diff))) for _ in range(5000)]
        dh = diff[hidden]
        print(f"{name:36s}{np.median(diff):+6.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]   hidden only: {np.median(dh):+.2f} (n={len(dh)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("masked_path")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.clip, a.masked_path, a.side)
