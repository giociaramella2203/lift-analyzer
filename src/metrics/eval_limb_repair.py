"""Oracle repair of the hidden knee: limb-length geometry vs plain interpolation vs doing nothing.

Usage (from the repository root):
  python src/metrics/eval_limb_repair.py <clip> [--side left]
Needs outputs/<clip>_rtm/h36m_2d.npz (RTMPose, with hidden_frames) and labels/<clip>_*.csv.

As in eval_masking.py, the knee is treated as lost in a window of +-K frames around each frame I marked as hidden
(THIS USES MY LABELS, so it is an upper bound on a repair, not a usable method; detect-then-repair comes later).
Repairs inside the window:
  interp   knee x,y linearly interpolated from frames outside the window
  limb     knee placed where the circle around the hip (radius = median thigh length) meets the circle around the ankle
           (radius = median shank length); of the two intersections, the one on the side the knee normally bends to.
           The lengths are medians over frames OUTSIDE the windows. If the circles do not meet, the knee goes on the
           hip-ankle line at the ratio of the two lengths.
Errors on frames I labelled:
  knee position error: distance to my knee click, in units of the median torso length
  hip-angle error: shoulder-hip-knee against my labels, degrees, mean bias removed (leave-one-out), as elsewhere
Bootstrap intervals are over labelled frames (few of them), so read them before the medians.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle                     # noqa: E402
from eval_labels import load_labels, J          # noqa: E402
from world3d import debiased_abs_err            # noqa: E402
from eval_masking import window                 # noqa: E402

H36M = {"left": dict(shoulder=11, hip=4, knee=5, ankle=6), "right": dict(shoulder=14, hip=1, knee=2, ankle=3)}


def cross(a, b):
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def limb_knee(hip, ankle, l1, l2, sign):
    """Knee from hip and ankle given thigh length l1 and shank length l2; sign = side of the hip-ankle line."""
    d = ankle - hip
    D = np.linalg.norm(d, axis=1) + 1e-9
    u = d / D[:, None]
    n = np.stack([-u[:, 1], u[:, 0]], 1)                      # unit normal; cross(u, n) > 0
    a = (l1 ** 2 - l2 ** 2 + D ** 2) / (2 * D)               # distance from hip along the line
    h2 = l1 ** 2 - a ** 2
    h = np.sqrt(np.clip(h2, 0, None))
    out_of_reach = h2 < 0
    a = np.where(out_of_reach, l1 / (l1 + l2) * D, a)         # circles do not meet: stay on the line
    return hip + a[:, None] * u + (sign * h)[:, None] * n


def repair_limb(kp, idx, m, hidden_frames_mask_for_stats):
    hip, knee, ankle = kp[:, idx["hip"]], kp[:, idx["knee"]], kp[:, idx["ankle"]]
    ok = ~hidden_frames_mask_for_stats
    l1 = np.median(np.linalg.norm(hip[ok] - knee[ok], axis=1))
    l2 = np.median(np.linalg.norm(knee[ok] - ankle[ok], axis=1))
    s = np.sign(cross(ankle[ok] - hip[ok], knee[ok] - hip[ok]))
    sign = 1.0 if np.median(s) >= 0 else -1.0
    y = kp.copy()
    k = limb_knee(hip[m], ankle[m], l1, l2, sign)
    y[m, idx["knee"]] = k
    return y, (l1, l2, sign)


def main(clip, side):
    idx = H36M[side]
    z = np.load(Path("outputs") / f"{clip}_rtm" / "h36m_2d.npz")
    d2, hid = z["kp2d"].astype(float), z["hidden_frames"]
    T = len(d2)
    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    gt = gt[gt.frame < T]
    fr = gt.frame.to_numpy()
    hidden = gt.knee_hidden.to_numpy() == 1
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ag = angle(S["shoulder"], S["hip"], S["knee"])
    torso = float(np.median(np.linalg.norm(d2[:, idx["shoulder"]] - d2[:, idx["hip"]], axis=1)))
    t = np.arange(T)

    def evaluate(y):
        kn = np.linalg.norm(y[fr, idx["knee"]] - S["knee"], axis=1) / torso
        ang = debiased_abs_err(angle(y[fr, idx["shoulder"]], y[fr, idx["hip"]], y[fr, idx["knee"]]), ag)
        return kn, ang

    res = {"RTMPose, no repair": evaluate(d2)}
    for K in (5, 15):
        m = window(T, hid, K)
        yi = d2.copy()
        for c in (0, 1):
            yi[m, idx["knee"], c] = np.interp(t[m], t[~m], d2[~m, idx["knee"], c])
        res[f"interp K{K}"] = evaluate(yi)
        yl, (l1, l2, sg) = repair_limb(d2, idx, m, m)
        res[f"limb-length K{K}"] = evaluate(yl)
        print(f"K{K}: thigh {l1:.0f} px, shank {l2:.0f} px, bend side {sg:+.0f}, repaired frames {int(m.sum())}")

    med = lambda e: f"{np.median(e):5.2f}" if len(e) else "  n/a"
    print(f"\n{clip}: {len(fr)} labelled frames ({int(hidden.sum())} knee-hidden), torso {torso:.0f} px")
    print(f"\n{'':22s}{'knee err (torso units), median':>34s}{'hip-angle err (deg), median':>34s}")
    print(f"{'':22s}{'all':>8s}{'hidden':>9s}{'visible':>9s}{'':>6s}{'all':>8s}{'hidden':>9s}{'visible':>9s}")
    for name, (kn, ang) in res.items():
        print(f"{name:22s}{med(kn):>8s}{med(kn[hidden]):>9s}{med(kn[~hidden]):>9s}{'':>6s}{med(ang):>8s}{med(ang[hidden]):>9s}{med(ang[~hidden]):>9s}")

    rng = np.random.default_rng(0)
    base_kn, base_ang = res["RTMPose, no repair"]
    print("\nPaired difference to 'no repair' on HIDDEN frames (negative = better), median [95% bootstrap]")
    for name, (kn, ang) in res.items():
        if name.startswith("RTMPose"):
            continue
        out = []
        for lab, a, b in (("knee err", kn, base_kn), ("angle err", ang, base_ang)):
            dh = (a - b)[hidden]
            bs = [np.median(rng.choice(dh, len(dh))) for _ in range(5000)]
            out.append(f"{lab} {np.median(dh):+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")
        print(f"{name:22s}" + "   ".join(out) + f"   (n={int(hidden.sum())})")
    print("\nPaired difference limb-length minus interp on HIDDEN frames, same K")
    for K in (5, 15):
        for lab, i in (("knee err", 0), ("angle err", 1)):
            dh = (res[f"limb-length K{K}"][i] - res[f"interp K{K}"][i])[hidden]
            bs = [np.median(rng.choice(dh, len(dh))) for _ in range(5000)]
            print(f"  K{K} {lab}: {np.median(dh):+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.clip, a.side)
