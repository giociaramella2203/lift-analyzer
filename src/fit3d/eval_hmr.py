"""Does a 3D-body-prior model (HMR2.0, 4D-Humans) change its hip angle less than RTMPose when the disc covers the knee?

Usage (from the repository root, after running notebooks/hmr2_fit3d.ipynb in Colab and saving its output):
  python src/fit3d/eval_hmr.py            # reads outputs/fit3d_hmr/hmr2_keypoints.npz

Same frames for both models (held-out subjects, camera 60457274, every 12th video frame). For each model and posture bin
(hip angle of RTMPose on the clean frame, the same bins as before) it prints the median change of the 2D hip angle caused by the disc
(disc version minus the same model's clean version), the share of frames changing by more than 10 degrees, and the knee shift in
torso lengths.  95% intervals over subjects.  Also printed: how well each model's CLEAN hip angle matches RTMPose's, as a sanity check
that the HMR2.0 joints and the left/right choice are right (the side is the one whose clean knee is closest to the true knee).
HMR2.0 joints are the first 25 of its 44 outputs, OpenPose order: shoulders 2/5, hips 9/12, knees 10/13 (right/left).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import angle, boot, fmt, hip_angle, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
HMR = {"right": dict(shoulder=2, hip=9, knee=10), "left": dict(shoulder=5, hip=12, knee=13)}


def main(cam="60457274"):
    d = np.load("outputs/fit3d_hmr/hmr2_keypoints.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    rec = {}
    for subj in sorted({n.split("/")[0] for n in names}):
        f, z = load(Path(f"outputs/fit3d/{subj}_{cam}.npz")), np.load(f"outputs/fit3d/{subj}_{cam}.npz")
        rows = {int(fr): i for i, fr in enumerate(z["frames"])}
        sel = [(k, rows[int(n.split("/")[1])]) for k, n in enumerate(names) if n.startswith(subj + "/")]
        k_h, k_r = np.array([a for a, _ in sel]), np.array([b for _, b in sel])
        gt_knee = f["gt_knee"][k_r]
        cl, hd = d["clean"][k_h], d["hidden"][k_h]
        dist = {s: np.median(np.linalg.norm(cl[:, HMR[s]["knee"]] - gt_knee, axis=1)) for s in HMR}
        s = min(dist, key=dist.get)
        g = {j: HMR[s][j] for j in ("shoulder", "hip", "knee")}
        pts = lambda a: {j: a[:, g[j]] for j in g}
        rec[subj] = dict(
            ref=hip_angle({j: f["rtm_clean"][j][k_r] for j in g}), torso=f["torso"][k_r],
            rtm_c={j: f["rtm_clean"][j][k_r] for j in g}, rtm_h={j: f["rtm_hidden"][j][k_r] for j in g},
            hmr_c=pts(cl), hmr_h=pts(hd))
    print(f"{len(rec)} subjects, {sum(len(r['ref']) for r in rec.values())} frames\n")

    ang = lambda p: angle(p["shoulder"], p["hip"], p["knee"])
    sd = np.concatenate([np.abs(ang(r["hmr_c"]) - ang(r["rtm_c"])) for r in rec.values()])
    print(f"sanity: |clean hip angle HMR2.0 - RTMPose|, median {np.median(sd):.1f} deg, 90th percentile {np.percentile(sd, 90):.1f} deg\n")
    print(f"{'model':10s}{'clean hip angle':>16s}{'frames':>8s}  {'hip-angle change, deg':26s}{'> 10 deg':>10s}{'knee shift / torso':>20s}")
    for model, c, h in (("RTMPose", "rtm_c", "rtm_h"), ("HMR2.0", "hmr_c", "hmr_h")):
        for lo, hi in BINS:
            def pick(fn):
                return {s: np.where((r["ref"] >= lo) & (r["ref"] < hi), fn(r), np.nan) for s, r in rec.items()}
            n = int(sum(((r["ref"] >= lo) & (r["ref"] < hi)).sum() for r in rec.values()))
            ch = boot(pick(lambda r: ang(r[h]) - ang(r[c])), np.median)
            big = boot(pick(lambda r: np.abs(ang(r[h]) - ang(r[c]))), lambda v: 100 * (v > 10).mean())
            ks = boot(pick(lambda r: np.linalg.norm(r[h]["knee"] - r[c]["knee"], axis=1) / r["torso"]), np.median)
            print(f"{model:10s}{f'{lo}-{hi} deg':>16s}{n:8d}  {fmt(ch):26s}{big[0]:9.0f}%{ks[0]:20.2f}")
        print()


if __name__ == "__main__":
    main()
