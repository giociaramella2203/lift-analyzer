"""Does the disc over the knee also cover OTHER joints, and does that explain the drift of the shoulder and hip?  No model is re-run.

Usage (from the repository root):
  python src/fit3d/eval_cover_all.py

Uses the true projected joints (all 25 Fit3D joints) and the disc geometry saved by run_pose.py.
Joint numbers (names are my reading of the overlay, indices are what matters): 0 pelvis, 1/4 hips, 2/5 knees, 3/6 ankles,
7 spine, 8 neck, 9 face, 10 head, 11/14 shoulders, 12/15 elbows, 13/16 wrists, 17-20 feet, 21-24 hands.
Table 1: per posture bin, the share of frames in which each joint (other than the occluded knee itself) lies inside the disc.
Table 2: per posture bin, frames split by whether at least one OTHER joint is covered; RTMPose shoulder shift, hip shift
         (torso lengths, medians) and the full hip-angle change (degrees, median), 95% intervals over subjects.
Reading guide
  * the groups with another joint covered show clearly larger shoulder/hip shifts and angle change -> hiding more of the body explains part of the drift
  * both groups similar                                                                               -> it is not about other covered joints
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import GT, angle, boot, by_subject, fmt, hip_angle, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))


def main():
    paths = sorted(p for p in Path("outputs/fit3d").glob("s*_*.npz") if not p.stem.endswith("_test"))
    F = []
    for p in paths:
        f, z = load(p), np.load(p)
        kn = GT[str(z["side"])]["knee"]
        cov = np.linalg.norm(z["gt2d"] - z["disc_c"][:, None, :], axis=2) < z["disc_r"][:, None]
        cov[:, kn] = False
        f["cov"] = cov
        f["ref"] = hip_angle(f["rtm_clean"])
        F.append(f)
    print(f"{len(F)} files, {sum(f['n'] for f in F)} frames per version\n")

    print("1. Share of frames in which each other joint lies inside the disc (percent; joints never covered are left out)")
    for lo, hi in BINS:
        sel = [(f["ref"] >= lo) & (f["ref"] < hi) for f in F]
        n = sum(int(s.sum()) for s in sel)
        share = np.sum([f["cov"][s].sum(axis=0) for f, s in zip(F, sel)], axis=0) / n * 100
        any_ = sum(int(f["cov"][s].any(axis=1).sum()) for f, s in zip(F, sel)) / n * 100
        items = ", ".join(f"{j}: {share[j]:.0f}%" for j in np.argsort(-share) if share[j] >= 1)
        print(f"  {lo}-{hi} deg ({n} frames): any other joint covered {any_:.0f}%   [{items or 'none'}]")

    print("\n2. Frames split by whether at least one OTHER joint is covered (RTMPose; medians, 95% intervals over subjects)")
    print(f"{'clean hip angle':>16s}{'other joint covered':>21s}{'frames':>8s}  {'shoulder shift':18s}{'hip shift':18s}{'full hip-angle change':26s}")
    for lo, hi in BINS:
        for flag in (False, True):
            def sel(f):
                return (f["ref"] >= lo) & (f["ref"] < hi) & (f["cov"].any(axis=1) == flag)
            n = int(sum(sel(f).sum() for f in F))
            if n < 20:
                print(f"{f'{lo}-{hi} deg':>16s}{'yes' if flag else 'no':>21s}{n:8d}  too few frames")
                continue

            def shift(f, j):
                return np.where(sel(f), np.linalg.norm(f["rtm_hidden"][j] - f["rtm_clean"][j], axis=1) / f["torso"], np.nan)
            ss = boot(by_subject(F, lambda f: shift(f, "shoulder")), np.median)
            hs = boot(by_subject(F, lambda f: shift(f, "hip")), np.median)
            fu = boot(by_subject(F, lambda f: np.where(sel(f), hip_angle(f["rtm_hidden"]) - f["ref"], np.nan)), np.median)
            print(f"{f'{lo}-{hi} deg':>16s}{'yes' if flag else 'no':>21s}{n:8d}  "
                  f"{ss[0]:5.2f} [{ss[1]:.2f},{ss[2]:.2f}]  {hs[0]:5.2f} [{hs[1]:.2f},{hs[2]:.2f}]  {fmt(fu):26s}")


if __name__ == "__main__":
    main()
