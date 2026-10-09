"""Label-free signals that might flag "the knee is hidden", checked against my hidden-knee labels.

Usage (from the repository root):
  python src/metrics/occlusion_signals.py <clip> [--side left]
Needs outputs/<clip>/mediapipe_keypoints.npz (MediaPipe), outputs/<clip>_rtm/mediapipe_keypoints.npz (RTMPose),
outputs/<clip>_rtm/h36m_2d.npz (RTMPose scores) and labels/<clip>_*.csv.

Signals, computed for every frame without using labels (higher = more suspicious):
  disagree   distance between the MediaPipe knee and the RTMPose knee, in units of the median torso length
  bone_rtm   how far the RTMPose thigh and shank lengths are from their own clip median (relative, summed)
  bone_mp    the same for MediaPipe
  path_dev   distance of the RTMPose knee from a 15-frame running median of its own path (torso units)
  speed      frame-to-frame RTMPose knee displacement (torso units)
  low_conf   minus the RTMPose knee score
AUC = probability that a random hidden-knee labelled frame scores higher than a random visible one (0.5 = chance).
Computed on the frames I labelled only, so n is small: read the bootstrap interval, not the point value.
Also reported on ascent frames only, because the plate hides the knee mostly during the ascent and a signal that
just detects "the bar is moving" would otherwise look good.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import LM                        # noqa: E402
from eval_labels import load_labels             # noqa: E402

H36M_KNEE = {"left": 5, "right": 2}


def auc(pos, neg):
    """Mann-Whitney AUC; ties count one half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def boot_auc(x, y, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if y[i].all() or (~y[i]).all():
            continue
        vals.append(auc(x[i][y[i]], x[i][~y[i]]))
    return np.percentile(vals, [2.5, 97.5])


def compute_signals(clip, side="left"):
    """Return dict name -> array (T,), plus the RTMPose knee path (T,2) and the median torso length."""
    mp = np.load(Path("outputs") / clip / "mediapipe_keypoints.npz")["keypoints"]
    rtm = np.load(Path("outputs") / f"{clip}_rtm" / "mediapipe_keypoints.npz")["keypoints"]
    score = np.load(Path("outputs") / f"{clip}_rtm" / "h36m_2d.npz")["score"][:, H36M_KNEE[side]]
    T = min(len(mp), len(rtm), len(score))
    mp, rtm, score = mp[:T], rtm[:T], score[:T]
    L = LM[side]
    P = lambda k, j: k[:, L[j], :2]
    torso = float(np.median(np.linalg.norm(P(rtm, "shoulder") - P(rtm, "hip"), axis=1)))

    def bones(k):
        th = np.linalg.norm(P(k, "hip") - P(k, "knee"), axis=1)
        sh = np.linalg.norm(P(k, "knee") - P(k, "ankle"), axis=1)
        return np.abs(th / np.median(th) - 1) + np.abs(sh / np.median(sh) - 1)

    kr = P(rtm, "knee")
    med = np.stack([median_filter(kr[:, c], size=15, mode="nearest") for c in (0, 1)], 1)
    speed = np.r_[0, np.linalg.norm(np.diff(kr, axis=0), axis=1)] / torso
    sig = dict(
        disagree=np.linalg.norm(P(mp, "knee") - kr, axis=1) / torso,
        bone_rtm=bones(rtm),
        bone_mp=bones(mp),
        path_dev=np.linalg.norm(kr - med, axis=1) / torso,
        speed=speed,
        low_conf=-score,
    )
    return sig, kr, torso


def main(clip, side):
    sig, _, torso = compute_signals(clip, side)
    T = len(next(iter(sig.values())))
    df = load_labels(clip)
    g = df.groupby("frame").agg(hidden=("knee_hidden", "max"), phase=("phase", "first")).reset_index()
    g = g[g.frame < T]
    fr = g.frame.to_numpy()
    y = g.hidden.to_numpy() == 1
    asc = (g.phase == "ascent").to_numpy()
    print(f"{clip}: {len(fr)} labelled frames, {int(y.sum())} knee-hidden; ascent frames {int(asc.sum())} "
          f"({int(y[asc].sum())} hidden); median torso {torso:.0f} px")
    print(f"\n{'signal':10s}{'AUC all':>9s}{'95% CI':>16s}{'AUC ascent only':>18s}{'n hid/vis (ascent)':>20s}")
    for name, v in sig.items():
        x = v[fr]
        ok = ~np.isnan(x)                       # frames where the signal exists (e.g. MediaPipe may lack the ankle)
        if ok.sum() < 5 or y[ok].all() or (~y[ok]).all():
            print(f"{name:10s}{'n/a':>9s}   (signal missing on {int((~ok).sum())} of {len(x)} labelled frames)")
            continue
        a = auc(x[ok & y], x[ok & ~y])
        lo, hi = boot_auc(x[ok], y[ok])
        m = ok & asc
        aa = auc(x[m & y], x[m & ~y])
        note = f"  (missing on {int((~ok).sum())})" if (~ok).any() else ""
        print(f"{name:10s}{a:9.2f}{f'[{lo:.2f}, {hi:.2f}]':>16s}{aa:18.2f}{f'{int((m & y).sum())}/{int((m & ~y).sum())}':>20s}{note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    a = ap.parse_args()
    main(a.clip, a.side)
