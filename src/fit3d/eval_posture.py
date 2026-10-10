"""Why does the disc matter mainly when the lifter is bent over?  Three checks on the saved Fit3D predictions (no model is re-run).

Usage (from the repository root):
  python src/fit3d/eval_posture.py

Posture bins = hip angle of the model on the CLEAN frame (same bins as eval_occlusion.py).
Per bin and model it prints:
  knee shift   how far the disc moves the model's knee, in torso lengths (median).
               Question: is the knee placed worse when bent over?  (the "harder to infer" idea)
  knee-only    change in hip angle if ONLY the knee is replaced by its hidden-version position, shoulder and hip kept from the
               clean frame (median).  Question: does the knee alone explain the angle change, or do shoulder/hip move too?
  full         the full change in hip angle (as in eval_occlusion.py, median).
  geometry     change in hip angle when the CLEAN knee is moved by a fixed 0.10 torso in 8 directions (mean |change|).
               Same knee error everywhere, so any difference between bins is pure geometry.  (the "angle is more sensitive" idea)
Reading guide
  * knee shift similar across bins, geometry much larger when bent  -> mostly geometry (angle sensitivity)
  * knee shift much larger when bent, geometry similar              -> mostly the model placing the hidden knee worse
  * both differ                                                     -> a mix
Intervals are 95% bootstrap intervals over subjects.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import MODELS, NAMES, angle, boot, by_subject, fmt, hip_angle, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
STEP = 0.10
DIRS = [np.array([np.cos(a), np.sin(a)]) for a in np.arange(8) * np.pi / 4]


def main():
    paths = sorted(p for p in Path("outputs/fit3d").glob("s*_*.npz") if not p.stem.endswith("_test"))
    F = [load(p) for p in paths]
    print(f"{len(F)} files, {sum(f['n'] for f in F)} frames per version\n")
    print(f"{'model':10s}{'clean hip angle':>16s}{'frames':>8s}  {'knee shift / torso':22s}{'knee-only change, deg':26s}{'full change, deg':26s}{'geometry |change|, deg':>24s}")
    for m in MODELS:
        for lo, hi in BINS:
            def pick(f, fn, lo=lo, hi=hi, m=m):
                ref = hip_angle(f[f"{m}_clean"])
                return np.where((ref >= lo) & (ref < hi), fn(f, m), np.nan)

            def shift(f, m):
                return np.linalg.norm(f[f"{m}_hidden"]["knee"] - f[f"{m}_clean"]["knee"], axis=1) / f["torso"]

            def knee_only(f, m):
                c, h = f[f"{m}_clean"], f[f"{m}_hidden"]
                return angle(c["shoulder"], c["hip"], h["knee"]) - hip_angle(c)

            def full(f, m):
                return hip_angle(f[f"{m}_hidden"]) - hip_angle(f[f"{m}_clean"])

            def geometry(f, m):
                c = f[f"{m}_clean"]
                base = hip_angle(c)
                ch = [np.abs(angle(c["shoulder"], c["hip"], c["knee"] + STEP * f["torso"][:, None] * d) - base) for d in DIRS]
                return np.mean(ch, axis=0)

            S = by_subject(F, lambda f: pick(f, shift))
            K = by_subject(F, lambda f: pick(f, knee_only))
            U = by_subject(F, lambda f: pick(f, full))
            G = by_subject(F, lambda f: pick(f, geometry))
            n = int(sum((~np.isnan(v)).sum() for v in S.values()))
            s, g = boot(S, np.median), boot(G, np.mean)
            print(f"{NAMES[m]:10s}{f'{lo}-{hi} deg':>16s}{n:8d}  {s[0]:5.2f} [{s[1]:.2f}, {s[2]:.2f}]     "
                  f"{fmt(boot(K, np.median)):26s}{fmt(boot(U, np.median)):26s}{g[0]:10.1f} [{g[1]:.1f}, {g[2]:.1f}]")
    print("\nknee-only vs full: if close, the knee explains the change; if full is larger, shoulder and hip also move behind the disc.")


if __name__ == "__main__":
    main()
