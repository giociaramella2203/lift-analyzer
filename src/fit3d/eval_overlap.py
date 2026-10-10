"""Does the disc over the knee also cover the hip when the lifter is bent over?  (Follow-up to eval_posture.py; no model is re-run.)

Usage (from the repository root):
  python src/fit3d/eval_overlap.py

Uses the TRUE projected joints (Fit3D) and the disc geometry saved by run_pose.py, plus the models' predictions.
Per model and posture bin (hip angle of the model on the clean frame) it prints:
  hip covered   share of frames where the true hip lies inside the disc (distance to the disc centre < radius)
  dist / radius median distance from the disc centre to the true hip, in disc radii (below 1 = inside)
  hip shift     how far the disc moves the model's hip, in torso lengths (median); shoulder shift likewise
  hip-only      change in hip angle if ONLY the hip is taken from the hidden version (shoulder and knee from the clean frame)
  knee-only     same for the knee only (as in eval_posture.py);   full = all three from the hidden version
Then, within each bin, frames split by whether the true hip is covered: hip shift and full change for each group.
Reading guide
  * hip covered often when bent over, rarely when upright, and the hip shifts more when covered -> the disc hiding the hip
    explains the extra change.  If the hip shifts just as much when NOT covered, the disc is not the reason (the model moves the
    hip for another reason, e.g. it loses the whole leg).
Intervals are 95% bootstrap intervals over subjects.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import GT, MODELS, NAMES, angle, boot, by_subject, fmt, hip_angle, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))


def main():
    paths = sorted(p for p in Path("outputs/fit3d").glob("s*_*.npz") if not p.stem.endswith("_test"))
    F = []
    for p in paths:
        f = load(p)
        z = np.load(p)
        g = GT[str(z["side"])]
        f["dist_hip"] = np.linalg.norm(z["gt2d"][:, g["hip"]] - z["disc_c"], axis=1) / z["disc_r"]
        f["dist_sh"] = np.linalg.norm(z["gt2d"][:, g["shoulder"]] - z["disc_c"], axis=1) / z["disc_r"]
        F.append(f)
    print(f"{len(F)} files, {sum(f['n'] for f in F)} frames per version\n")
    print(f"{'model':10s}{'clean hip angle':>16s}{'frames':>8s}{'hip covered':>13s}{'shoulder cov.':>14s}{'dist/radius':>13s}  "
          f"{'hip shift':16s}{'shoulder shift':16s}{'hip-only':22s}{'knee-only':22s}{'full':22s}")
    for m in MODELS:
        for lo, hi in BINS:
            def inbin(f):
                ref = hip_angle(f[f"{m}_clean"])
                return (ref >= lo) & (ref < hi)

            def pick(fn):
                return by_subject(F, lambda f: np.where(inbin(f), fn(f), np.nan))

            def shift(f, j):
                return np.linalg.norm(f[f"{m}_hidden"][j] - f[f"{m}_clean"][j], axis=1) / f["torso"]

            def only(f, j):
                c, h = f[f"{m}_clean"], f[f"{m}_hidden"]
                a = {k: c[k] for k in ("shoulder", "hip", "knee")}
                a[j] = h[j]
                return angle(a["shoulder"], a["hip"], a["knee"]) - hip_angle(c)

            n = int(sum(np.sum(inbin(f)) for f in F))
            cov = boot(pick(lambda f: (f["dist_hip"] < 1).astype(float)), np.mean)
            cov_s = boot(pick(lambda f: (f["dist_sh"] < 1).astype(float)), np.mean)
            dr = boot(pick(lambda f: f["dist_hip"]), np.median)
            hs, ss = boot(pick(lambda f: shift(f, "hip")), np.median), boot(pick(lambda f: shift(f, "shoulder")), np.median)
            ho, ko = boot(pick(lambda f: only(f, "hip")), np.median), boot(pick(lambda f: only(f, "knee")), np.median)
            fu = boot(pick(lambda f: hip_angle(f[f"{m}_hidden"]) - hip_angle(f[f"{m}_clean"])), np.median)
            print(f"{NAMES[m]:10s}{f'{lo}-{hi} deg':>16s}{n:8d}{100 * cov[0]:12.0f}%{100 * cov_s[0]:13.0f}%{dr[0]:13.2f}  "
                  f"{hs[0]:5.2f} [{hs[1]:.2f},{hs[2]:.2f}]  {ss[0]:5.2f} [{ss[1]:.2f},{ss[2]:.2f}]  "
                  f"{fmt(ho):22s}{fmt(ko):22s}{fmt(fu):22s}")

    print("\nSplit by whether the true hip is covered by the disc (hip shift in torso lengths, full change in degrees; medians)")
    print(f"{'model':10s}{'clean hip angle':>16s}{'hip':>10s}{'frames':>8s}  {'hip shift':20s}{'full change':24s}")
    for m in MODELS:
        for lo, hi in BINS:
            for covered in (True, False):
                def sel(f):
                    ref = hip_angle(f[f"{m}_clean"])
                    return (ref >= lo) & (ref < hi) & ((f["dist_hip"] < 1) == covered)
                n = int(sum(np.sum(sel(f)) for f in F))
                if n < 20:
                    print(f"{NAMES[m]:10s}{f'{lo}-{hi} deg':>16s}{'covered' if covered else 'not':>10s}{n:8d}  too few frames")
                    continue
                hs = boot(by_subject(F, lambda f: np.where(sel(f), np.linalg.norm(f[f'{m}_hidden']['hip'] - f[f'{m}_clean']['hip'], axis=1) / f["torso"], np.nan)), np.median)
                fu = boot(by_subject(F, lambda f: np.where(sel(f), hip_angle(f[f"{m}_hidden"]) - hip_angle(f[f"{m}_clean"]), np.nan)), np.median)
                print(f"{NAMES[m]:10s}{f'{lo}-{hi} deg':>16s}{'covered' if covered else 'not':>10s}{n:8d}  "
                      f"{hs[0]:5.2f} [{hs[1]:.2f}, {hs[2]:.2f}]  {fmt(fu):24s}")


if __name__ == "__main__":
    main()
