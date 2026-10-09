"""Evaluate the 2D-to-3D lifter output (from notebooks/lift3d_motionbert.ipynb) the same way as world3d.py.

Usage (from the repository root):
  python src/metrics/eval_lift3d.py <clip> <path to lift3d_<clip>.npz> [--key pose3d_conf|pose3d_noconf] [--side left]
Needs outputs/<clip>_rtm/h36m_2d.npz (the lifter's 2D input, from export_h36m.py) and labels/<clip>_*.csv.

Compares, on the labelled frames, the hip angle (shoulder-hip-knee) against your hand labels:
  RTMPose pixels (2D)  |  lifted x,y only  |  lifted x,y,z
and prints 3D bone-length stability and the frame-to-frame change of the angle over all frames.
The lifter output is in normalised units and its axes follow MotionBERT's convention; I assume x,y are the image-plane
axes and z is depth. Your labels are 2D, so this is not an accuracy measurement of the 3D (see README, section 6).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle                      # noqa: E402
from eval_labels import load_labels, J           # noqa: E402
from world3d import debiased_abs_err, summary    # noqa: E402

H36M = {"left": dict(shoulder=11, hip=4, knee=5, ankle=6), "right": dict(shoulder=14, hip=1, knee=2, ankle=3)}


def main(clip, lift_path, key, side):
    idx = H36M[side]
    p3 = np.load(lift_path)[key]
    d2 = np.load(Path("outputs") / f"{clip}_rtm" / "h36m_2d.npz")["kp2d"]
    T = min(len(p3), len(d2))
    p3, d2 = p3[:T], d2[:T]
    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    gt = gt[gt.frame < T]
    fr = gt.frame.to_numpy()
    hidden = gt.knee_hidden.to_numpy() == 1
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ag = angle(S["shoulder"], S["hip"], S["knee"])
    A = lambda arr, dims: angle(arr[:, idx["shoulder"], :dims], arr[:, idx["hip"], :dims], arr[:, idx["knee"], :dims])
    print(f"{clip}, {key}, side={side}: {len(fr)} labelled frames ({int(hidden.sum())} knee-hidden)")

    print("\n1) 3D bone length, relative deviation from the clip median, %")
    for name, (a, b) in {"thigh": ("hip", "knee"), "shank": ("knee", "ankle"), "torso": ("shoulder", "hip")}.items():
        L = np.linalg.norm(p3[:, idx[a]] - p3[:, idx[b]], axis=1)
        dev = np.abs(L - np.median(L)) / np.median(L) * 100
        print(f"   {name:6s} all: {summary(dev)} | labelled hidden: {summary(dev[fr][hidden])} | visible: {summary(dev[fr][~hidden])}")

    print("\n2) Hip-angle error against your labels, degrees, mean bias removed (leave-one-out)")
    e2 = debiased_abs_err(A(d2[fr], 2), ag)
    rows = {"RTMPose pixels (2D)": e2,
            "lifted x,y only": debiased_abs_err(A(p3[fr], 2), ag),
            "lifted x,y,z (3D)": debiased_abs_err(A(p3[fr], 3), ag)}
    for name, e in rows.items():
        print(f"   {name:20s} all: {summary(e)} | hidden: {summary(e[hidden])} | visible: {summary(e[~hidden])}")

    rng = np.random.default_rng(0)
    print("\n3) Median of (lifted error minus RTMPose-2D error), 95% bootstrap interval")
    for name in ("lifted x,y only", "lifted x,y,z (3D)"):
        e = rows[name] - e2
        b = [np.median(rng.choice(e, len(e))) for _ in range(5000)]
        print(f"   {name:20s}: {np.median(e):+.2f} deg  [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")

    print("\n4) Frame-to-frame change of the hip angle, all frames (median and p90 of |change|, degrees)")
    for name, ang in (("RTMPose pixels (2D)", A(d2, 2)), ("lifted x,y", A(p3, 2)), ("lifted x,y,z", A(p3, 3))):
        dd = np.abs(np.diff(ang))
        print(f"   {name:20s}: median {np.median(dd):.2f}  p90 {np.percentile(dd, 90):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("lift_path")
    ap.add_argument("--key", default="pose3d_conf", choices=["pose3d_conf", "pose3d_noconf"])
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.clip, a.lift_path, a.key, a.side)
