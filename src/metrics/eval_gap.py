"""Occlusion-attributable hip-angle error: signed error on hidden frames minus signed error on visible frames.

Usage (from the repository root):
  python src/metrics/eval_gap.py <clip> [--side left] [--sessions A B]   (default: all sessions, averaged)
Why: the model's hip angle differs from my clicks by a roughly constant amount (a convention gap: where the model puts the
joint vs where I click). The bias-removed error used elsewhere subtracts the mean over ALL labelled frames, which mixes that
constant with the damage on hidden frames. Here the constant is read from the VISIBLE frames and the hidden frames are
compared with it: gap = median(model - label | hidden) - median(model - label | visible), in degrees. Zero means the hidden
frames are no worse than the visible ones, apart from the constant. Bootstrap 95% interval over labelled frames.
Same oracle windows and repairs as eval_limb_repair.py (the repairs use my hidden labels to place the window: upper bound).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle                                         # noqa: E402
from eval_labels import load_labels, J                              # noqa: E402
from eval_masking import window                                     # noqa: E402
from eval_limb_repair import H36M, repair_limb                      # noqa: E402


def main(clip, side, sessions=None):
    idx = H36M[side]
    z = np.load(Path("outputs") / f"{clip}_rtm" / "h36m_2d.npz")
    d2, hid = z["kp2d"].astype(float), z["hidden_frames"]
    T = len(d2)
    df = load_labels(clip)
    if sessions:
        df = df[df.session.isin(sessions)]
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    g = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    g = g[g.frame < T]
    fr, hidden = g.frame.to_numpy(), g.knee_hidden.to_numpy() == 1
    S = {j: g[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ag = angle(S["shoulder"], S["hip"], S["knee"])
    t = np.arange(T)

    def signed(y):
        return angle(y[fr, idx["shoulder"]], y[fr, idx["hip"]], y[fr, idx["knee"]]) - ag

    variants = {"RTMPose, no repair": d2}
    for K in (5, 15):
        m = window(T, hid, K)
        yi = d2.copy()
        for c in (0, 1):
            yi[m, idx["knee"], c] = np.interp(t[m], t[~m], d2[~m, idx["knee"], c])
        variants[f"interp K{K}"] = yi
        variants[f"limb-length K{K}"] = repair_limb(d2, idx, m, m)[0]

    rng = np.random.default_rng(0)
    print(f"{clip}: {int(hidden.sum())} hidden / {int((~hidden).sum())} visible labelled frames (signed errors, degrees)")
    print(f"{'':22s}{'visible med':>12s}{'hidden med':>12s}{'gap':>8s}   95% interval of gap")
    for name, y in variants.items():
        e = signed(y)
        x, v = e[hidden], e[~hidden]
        bs = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(v, len(v))) for _ in range(5000)]
        print(f"{name:22s}{np.median(v):+12.1f}{np.median(x):+12.1f}{np.median(x) - np.median(v):+8.1f}   [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--sessions", nargs="*", help="use only these label sessions (default: all, averaged)")
    a = ap.parse_args()
    main(a.clip, a.side, a.sessions)
