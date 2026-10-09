"""Detect-then-repair WITHOUT using hidden labels at test time.

Usage (from the repository root):
  python src/metrics/eval_detect_repair.py <train_clip> <test_clip> [--signal disagree] [--K 5] [--side left]
Pipeline: flag frames where a label-free signal (see occlusion_signals.py) exceeds a threshold, widen each flag by +-K frames,
replace the RTMPose knee in flagged frames by linear interpolation from unflagged frames, recompute the hip angle.
The threshold is tuned on <train_clip> (maximum of true-positive rate minus false-positive rate on the frames I labelled
there) and then applied unchanged to <test_clip>. Run it in both directions to see how fragile the threshold is.

Reports on <test_clip>: how many labelled hidden / visible frames were flagged, and knee-position error (torso units) and
hip-angle error (deg, mean bias removed leave-one-out) with and without the repair, on ALL labelled frames (a real system
does not know which frames are hidden, so a repair that damages visible frames counts against it) and on hidden frames.
Few labelled frames: read the bootstrap interval of the paired difference (repair minus no repair; negative = better).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle                         # noqa: E402
from eval_labels import load_labels, J              # noqa: E402
from world3d import debiased_abs_err                # noqa: E402
from occlusion_signals import compute_signals       # noqa: E402

H36M = {"left": dict(shoulder=11, hip=4, knee=5), "right": dict(shoulder=14, hip=1, knee=2)}


def labelled(clip, T):
    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    gt = gt[gt.frame < T]
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    return gt.frame.to_numpy(), gt.knee_hidden.to_numpy() == 1, S


def tune(x, y):
    ok = ~np.isnan(x)
    best, thr = -9, None
    for t in np.unique(x[ok]):
        flag = ok & (x >= t)
        j = flag[y].mean() - flag[~y].mean()
        if j > best:
            best, thr = j, t
    return thr, best


def dilate(flag, K):
    out = np.zeros_like(flag)
    for i in np.flatnonzero(flag):
        out[max(0, i - K):i + K + 1] = True
    return out


def main(train, test, signal, K, side):
    idx = H36M[side]
    sig_tr, _, _ = compute_signals(train, side)
    Ttr = len(sig_tr[signal])
    fr_tr, hid_tr, _ = labelled(train, Ttr)
    thr, j = tune(sig_tr[signal][fr_tr], hid_tr)
    print(f"threshold for '{signal}' tuned on {train}: {thr:.3f} (TPR-FPR = {j:.2f} on {len(fr_tr)} labelled frames)")

    sig, _, _ = compute_signals(test, side)
    x = sig[signal]
    T = len(x)
    z = np.load(Path("outputs") / f"{test}_rtm" / "h36m_2d.npz")
    d2 = z["kp2d"][:T].astype(float)
    fr, hidden, S = labelled(test, T)
    torso = float(np.median(np.linalg.norm(d2[:, idx["shoulder"]] - d2[:, idx["hip"]], axis=1)))
    raw_flag = np.nan_to_num(x, nan=-1e9) >= thr
    flag = dilate(raw_flag, K)
    print(f"{test}: flagged {raw_flag.mean() * 100:.0f}% of all frames ({flag.mean() * 100:.0f}% after +-{K} widening)")
    lf = flag[fr]
    print(f"  on labelled frames: flagged {int(lf[hidden].sum())}/{int(hidden.sum())} hidden, {int(lf[~hidden].sum())}/{int((~hidden).sum())} visible")

    t = np.arange(T)
    y = d2.copy()
    if flag.any() and (~flag).any():
        for c in (0, 1):
            y[flag, idx["knee"], c] = np.interp(t[flag], t[~flag], d2[~flag, idx["knee"], c])

    ag = angle(S["shoulder"], S["hip"], S["knee"])

    def ev(k):
        kn = np.linalg.norm(k[fr, idx["knee"]] - S["knee"], axis=1) / torso
        an = debiased_abs_err(angle(k[fr, idx["shoulder"]], k[fr, idx["hip"]], k[fr, idx["knee"]]), ag)
        return kn, an

    (k0, a0), (k1, a1) = ev(d2), ev(y)
    med = lambda e: np.median(e) if len(e) else float("nan")
    print(f"\n{'':18s}{'knee err all':>14s}{'hidden':>8s}{'angle err all':>15s}{'hidden':>8s}")
    print(f"{'no repair':18s}{med(k0):14.2f}{med(k0[hidden]):8.2f}{med(a0):15.2f}{med(a0[hidden]):8.2f}")
    print(f"{'detect + interp':18s}{med(k1):14.2f}{med(k1[hidden]):8.2f}{med(a1):15.2f}{med(a1[hidden]):8.2f}")
    rng = np.random.default_rng(0)
    print("\nPaired difference, repair minus no repair (negative = better), median [95% bootstrap]")
    for lab, d, mask in (("knee err, all", k1 - k0, slice(None)), ("angle err, all", a1 - a0, slice(None)),
                         ("angle err, hidden", (a1 - a0)[hidden], slice(None))):
        bs = [np.median(rng.choice(d, len(d))) for _ in range(5000)]
        print(f"  {lab:18s}{np.median(d):+6.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]  (n={len(d)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--signal", default="disagree")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.train, a.test, a.signal, a.K, a.side)
