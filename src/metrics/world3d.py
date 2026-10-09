"""Exploratory 3D check: do MediaPipe's 3D world landmarks help when the knee is hidden?

Usage: python src/metrics/world3d.py <clip> [--side left]
Run from the repository root. Needs outputs/<clip>/mediapipe_keypoints.npz and labels/<clip>_*.csv.

MediaPipe saves two sets of landmarks: pixel coordinates and "world" landmarks (metres, hip-centred,
estimated from the single image, so depth is a model guess). RTMPose has no 3D output here.

1. Bone-length stability: in 3D the thigh, shank and torso should keep a constant length. How much do they vary,
   on frames where you marked the knee as hidden vs visible?
2. Hip angle against your hand labels, three ways: image pixels, world x/y only, full world x/y/z.
   Each variant has its mean signed bias removed (leave-one-out) so only the random error is compared.
Your labels are 2D (image plane), so a 3D angle is compared with a 2D label angle: that is not a 3D ground truth.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import LM, angle          # noqa: E402
from eval_labels import load_labels, J   # noqa: E402


def summary(x):
    return f"median {np.median(x):5.2f}  p90 {np.percentile(x, 90):5.2f}" if len(x) else "n/a"


def debiased_abs_err(est, gt):
    err = est - gt
    out = np.empty_like(err)
    for k in range(len(err)):
        others = np.delete(err, k)
        out[k] = abs(err[k] - np.nanmean(others))
    return out


def main(clip, side):
    z = np.load(f"outputs/{clip}/mediapipe_keypoints.npz")
    kp, world = z["keypoints"], z["world"]
    idx = {j: LM[side][j] for j in ("shoulder", "hip", "knee", "ankle")}

    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    frames = gt.frame.to_numpy()
    hidden = gt.knee_hidden.to_numpy() == 1
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ang_gt = angle(S["shoulder"], S["hip"], S["knee"])
    print(f"{clip}: side={side}, {len(frames)} labelled frames ({int(hidden.sum())} knee-hidden, {int((~hidden).sum())} visible)")

    # 1. bone-length stability in world coordinates
    pairs = {"thigh (hip-knee)": ("hip", "knee"), "shank (knee-ankle)": ("knee", "ankle"), "torso (shoulder-hip)": ("shoulder", "hip")}
    print("\n1) 3D bone length, relative deviation from the clip median, in % (smaller = more stable)")
    for name, (a, b) in pairs.items():
        L = np.linalg.norm(world[:, idx[a], :] - world[:, idx[b], :], axis=1)
        dev = np.abs(L - np.nanmedian(L)) / np.nanmedian(L) * 100
        d = dev[frames]
        print(f"   {name:22s} all frames: {summary(dev[~np.isnan(dev)])} | labelled hidden: {summary(d[hidden])} | labelled visible: {summary(d[~hidden])}")

    # 2. hip angle vs hand labels
    img = angle(kp[frames][:, idx["shoulder"], :2], kp[frames][:, idx["hip"], :2], kp[frames][:, idx["knee"], :2])
    w = world[frames]
    wxy = angle(w[:, idx["shoulder"], :2], w[:, idx["hip"], :2], w[:, idx["knee"], :2])
    w3d = angle(w[:, idx["shoulder"], :], w[:, idx["hip"], :], w[:, idx["knee"], :])
    print("\n2) Hip-angle error against your labels, degrees, mean bias removed (leave-one-out)")
    for name, est in (("image pixels (2D)", img), ("world x,y only", wxy), ("world x,y,z (3D)", w3d)):
        e = debiased_abs_err(est, ang_gt)
        print(f"   {name:18s} all: {summary(e)} | hidden: {summary(e[hidden])} | visible: {summary(e[~hidden])}")
    print(f"\n   mean 3D minus 2D-pixel angle: {np.nanmean(w3d - img):+.1f} deg; median |difference|: {np.nanmedian(np.abs(w3d - img)):.1f} deg")
    zs = world[:, idx["knee"], 2] - world[:, idx["hip"], 2]
    print(f"   knee depth relative to hip (z, m): median {np.nanmedian(zs):+.3f}, std {np.nanstd(zs):.3f}")
    extra_checks(clip, side, kp, world, frames, ang_gt, idx)


def extra_checks(clip, side, kp, world, frames, ang_gt, idx):
    """Is the 3D-vs-2D gap larger than chance, and is the 3D angle noisier from frame to frame?"""
    rng = np.random.default_rng(0)
    A = lambda arr, d: angle(arr[:, idx["shoulder"], :d], arr[:, idx["hip"], :d], arr[:, idx["knee"], :d])
    e2 = debiased_abs_err(A(kp[frames], 2), ang_gt)
    print("\n3) Is the gap bigger than chance? Median of (3D error minus 2D-pixel error) over frames, 95% bootstrap interval")
    for name, arr, d in (("world x,y,z", world, 3), ("world x,y only", world, 2)):
        e = debiased_abs_err(A(arr[frames], d), ang_gt) - e2
        b = [np.median(rng.choice(e, len(e))) for _ in range(5000)]
        print(f"   {name:15s}: {np.median(e):+.2f} deg  [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]")
    ok = ~np.isnan(world[:, idx["hip"], 0])
    print("\n4) Frame-to-frame change of the hip angle over ALL frames (degrees, median and p90 of |change|)")
    for name, ang in (("pixel 2D", A(kp, 2)), ("world x,y", A(world, 2)), ("world x,y,z", A(world, 3))):
        d = np.abs(np.diff(ang[ok]))
        print(f"   {name:12s}: median {np.median(d):.2f}  p90 {np.percentile(d, 90):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.clip, a.side)
