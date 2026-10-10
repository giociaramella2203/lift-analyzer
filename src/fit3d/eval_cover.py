"""Is the drift of the shoulder and hip caused by the disc also hiding the arms?  Step 2: evaluate the outputs of run_cover.py.

Usage (from the repository root):
  python src/fit3d/eval_cover.py

Three disc placements on the same frames (every 2nd processed frame), RTMPose, change relative to the same frame without a disc:
  knee, plate-sized   the original experiment ('hidden')
  knee, half-size     the knee is still hidden, fewer neighbouring joints are
  wrists, plate-sized the hands and forearms are hidden, the knee is NOT at the centre of the disc
For each placement and posture bin (hip angle of the model on the clean frame) it prints: how often the true knee is inside the disc,
how often at least one other joint is (from the true joints), then the median shift of the knee, shoulder and hip (torso lengths) and
the median change of the hip angle (degrees), 95% intervals over subjects.
Reading guide
  * wrists disc: shoulder/hip shift and angle change clearly above zero while the knee is not covered  -> hiding the arms alone is enough to disturb them
  * half-size knee disc: knee shift stays large but shoulder/hip shift and angle change fall            -> the extra drift comes from the neighbouring joints the big disc also hides
  * all three similar                                                                                   -> not about the arms
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import GT, boot, by_subject, fmt, hip_angle, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
PLACEMENTS = ("knee, plate-sized", "knee, half-size", "wrists, plate-sized")


def main():
    F = []
    for p in sorted(Path("outputs/fit3d").glob("s*_*.npz")):
        if p.stem.endswith("_test"):
            continue
        cp = p.with_name(f"cover_{p.stem}.npz")
        if not cp.exists():
            print("missing", cp.name)
            continue
        f, z, c = load(p), np.load(p), np.load(cp)
        i = c["idx"]
        kn = GT[str(z["side"])]["knee"]
        gt = z["gt2d"][i]
        g =dict(torso=f["torso"][i], subj=f["subj"], idx=i)
        for name in ("clean",):
            g[name] = {k: v[i] for k, v in f["rtm_clean"].items()}
        g["knee, plate-sized"] = {k: v[i] for k, v in f["rtm_hidden"].items()}
        # model side for the new runs: same side as load() chose for the original run (closest to the true knee on clean frames)
        side = min((s for s in ("left", "right")), key=lambda s: np.nanmedian(np.linalg.norm(
            z["rtm_clean"][:, {"left": 25, "right": 26}[s], :2] - f["gt_knee"], axis=1)))
        L = {"left": dict(shoulder=11, hip=23, knee=25), "right": dict(shoulder=12, hip=24, knee=26)}[side]
        for name, key in (("knee, half-size", "knee_half"), ("wrists, plate-sized", "wrists_full")):
            g[name] = {j: c[f"rtm_{key}"][:, L[j], :2] for j in ("shoulder", "hip", "knee")}
        discs = {"knee, plate-sized": (z["disc_c"][i], z["disc_r"][i]),
                 "knee, half-size": (c["centre_knee_half"], c["radius_knee_half"]),
                 "wrists, plate-sized": (c["centre_wrists_full"], c["radius_wrists_full"])}
        for name, (ctr, r) in discs.items():
            cov = np.linalg.norm(gt - ctr[:, None, :], axis=2) < r[:, None]
            g[f"knee_in_{name}"] = cov[:, kn].copy()
            cov[:, kn] = False
            g[f"other_in_{name}"] = cov.any(axis=1)
        g["ref"] = hip_angle(g["clean"])
        F.append(g)
    if not F:
        raise SystemExit("no cover files; run run_cover.py first")
    print(f"{len(F)} files, {sum(len(g['ref']) for g in F)} frames per placement\n")
    print(f"{'clean hip angle':>16s}{'frames':>8s}  {'disc placement':20s}{'knee covered':>13s}{'other covered':>14s}  "
          f"{'knee shift':12s}{'shoulder shift':16s}{'hip shift':16s}{'hip-angle change':24s}")
    for lo, hi in BINS:
        n = int(sum(((g["ref"] >= lo) & (g["ref"] < hi)).sum() for g in F))
        for name in PLACEMENTS:
            def pick(fn):
                return by_subject(F, lambda g: np.where((g["ref"] >= lo) & (g["ref"] < hi), fn(g), np.nan))

            def shift(g, j):
                return np.linalg.norm(g[name][j] - g["clean"][j], axis=1) / g["torso"]
            kc = 100 * np.mean(np.concatenate([g[f"knee_in_{name}"][(g["ref"] >= lo) & (g["ref"] < hi)] for g in F]))
            oc = 100 * np.mean(np.concatenate([g[f"other_in_{name}"][(g["ref"] >= lo) & (g["ref"] < hi)] for g in F]))
            ks, ss, hs = (boot(pick(lambda g, j=j: shift(g, j)), np.median)[0] for j in ("knee", "shoulder", "hip"))
            ch = boot(pick(lambda g: hip_angle(g[name]) - g["ref"]), np.median)
            print(f"{f'{lo}-{hi} deg':>16s}{n:8d}  {name:20s}{kc:12.0f}%{oc:13.0f}%  {ks:8.2f}    {ss:8.2f}        {hs:8.2f}        {fmt(ch):24s}")
        print()


if __name__ == "__main__":
    main()
