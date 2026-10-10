"""Why did the body-prior model (HMR2.0) not beat RTMPose under the disc?  Two offline tests on the saved outputs (no model is re-run).

Usage (from the repository root, after eval_hmr.py works):
  python src/fit3d/eval_hmr_why.py

Both models, same frames (held-out subjects s07-s11, camera 60457274, every 12th video frame), plate-sized disc on the knee.
Posture bins use RTMPose's hip angle on the clean frame, as before.  95% intervals over subjects.

Test A, "is the model's knee under the disc better than a simple guess?"
  A linear prior predicts the knee from the shoulder, hip and ankle of the SAME image (positions relative to the hip, in torso lengths).
  It is fitted on the clean frames of the other subjects (leave-one-subject-out), separately for each model, to reproduce that model's
  own clean knee.  Then, on the disc images, three numbers (torso lengths, median):
    model shift    distance between the model's knee on the disc image and on the clean image
    prior shift    distance between the prior's guess (from the disc-image shoulder/hip/ankle) and the model's clean knee
    prior floor    the same distance on CLEAN images: how well the prior can do when nothing is hidden
  Reading guide: model shift much larger than prior shift -> the model is worse than a guess from body proportions;
  about equal -> it behaves like a prior; much smaller -> it uses image evidence the prior does not have.
Test B, "does temporal context rescue it?"
  Synthetic sequences per subject: disc version in blocks of 3 samples out of every 9 (the same 0.7 s as before), clean elsewhere,
  at three offsets so every frame is hidden once.  Hip-angle change vs the same sequence without a disc, inside the blocks, without
  repair and with the knee interpolated linearly over the block frames (true block positions, an upper bound).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import LM, angle, boot, fmt, hip_angle, interp_knee, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
HMR = {"right": dict(shoulder=2, hip=9, knee=10, ankle=11), "left": dict(shoulder=5, hip=12, knee=13, ankle=14)}
RTM_ANKLE = {"left": 27, "right": 28}
J = ("shoulder", "hip", "knee", "ankle")
PERIOD, BLOCK = 9, 3


def load_all(cam="60457274"):
    d = np.load("outputs/fit3d_hmr/hmr2_keypoints.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    rec = {}
    for subj in sorted({n.split("/")[0] for n in names}):
        p = Path(f"outputs/fit3d/{subj}_{cam}.npz")
        f, z = load(p), np.load(p)
        rows = {int(fr): i for i, fr in enumerate(z["frames"])}
        sel = sorted((int(n.split("/")[1]), k) for k, n in enumerate(names) if n.startswith(subj + "/"))
        kh = np.array([k for _, k in sel])
        kr = np.array([rows[fr] for fr, _ in sel])
        gt = f["gt_knee"]
        # RTMPose side: closest to the true knee on the clean frames (as in load())
        sr = min(LM, key=lambda s: np.nanmedian(np.linalg.norm(z["rtm_clean"][:, LM[s]["knee"], :2] - gt, axis=1)))
        rtm = lambda a: {"shoulder": a[kr, LM[sr]["shoulder"], :2], "hip": a[kr, LM[sr]["hip"], :2],
                         "knee": a[kr, LM[sr]["knee"], :2], "ankle": a[kr, RTM_ANKLE[sr], :2]}
        cl, hd = d["clean"][kh], d["hidden"][kh]
        sh = min(HMR, key=lambda s: np.median(np.linalg.norm(cl[:, HMR[s]["knee"]] - gt[kr], axis=1)))
        hmr = lambda a: {j: a[:, HMR[sh][j]] for j in J}
        rec[subj] = dict(torso=f["torso"][kr], rtm=dict(c=rtm(z["rtm_clean"]), h=rtm(z["rtm_hidden"])),
                         hmr=dict(c=hmr(cl), h=hmr(hd)))
        rec[subj]["ref"] = hip_angle(rec[subj]["rtm"]["c"])
    return rec


def feats(p, torso):
    return np.hstack([(p["shoulder"] - p["hip"]) / torso[:, None], (p["ankle"] - p["hip"]) / torso[:, None], np.ones((len(torso), 1))])


def test_a(rec):
    print("Test A: the model's knee under the disc vs a linear guess from shoulder, hip and ankle (torso lengths, medians)")
    print(f"{'model':8s}{'clean hip angle':>16s}{'frames':>8s}{'model shift':>14s}{'prior shift':>13s}{'prior floor':>13s}")
    for m in ("rtm", "hmr"):
        out = {}
        for s in rec:
            tr = [t for t in rec if t != s]
            X = np.vstack([feats(rec[t][m]["c"], rec[t]["torso"]) for t in tr])
            Y = np.vstack([(rec[t][m]["c"]["knee"] - rec[t][m]["c"]["hip"]) / rec[t]["torso"][:, None] for t in tr])
            ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            X, Y = X[ok], Y[ok]
            W = np.linalg.lstsq(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ Y, rcond=None)[0]
            r = rec[s]
            guess = lambda p: p["hip"] + r["torso"][:, None] * (feats(p, r["torso"]) @ W)
            kc = r[m]["c"]["knee"]
            dist = lambda a: np.linalg.norm(a - kc, axis=1) / r["torso"]
            out[s] = (dist(r[m]["h"]["knee"]), dist(guess(r[m]["h"])), dist(guess(r[m]["c"])))
        for lo, hi in BINS:
            def pick(i):
                return {s: np.where((rec[s]["ref"] >= lo) & (rec[s]["ref"] < hi), out[s][i], np.nan) for s in rec}
            n = int(sum(((rec[s]["ref"] >= lo) & (rec[s]["ref"] < hi)).sum() for s in rec))
            v = [boot(pick(i), np.median)[0] for i in range(3)]
            print(f"{('RTMPose' if m == 'rtm' else 'HMR2.0'):8s}{f'{lo}-{hi} deg':>16s}{n:8d}{v[0]:14.2f}{v[1]:13.2f}{v[2]:13.2f}")
    print()


def test_b(rec):
    print(f"Test B: hip-angle change inside hidden blocks ({BLOCK} of every {PERIOD} samples), without and with linear interpolation of the knee")
    print(f"{'model':8s}{'clean hip angle':>16s}{'frames':>8s}  {'no repair':24s}{'interpolated knee':24s}")
    for m in ("rtm", "hmr"):
        acc = {}
        for s, r in rec.items():
            n = len(r["ref"])
            c, h = r[m]["c"], r[m]["h"]
            base = angle(c["shoulder"], c["hip"], c["knee"])
            for off in (0, BLOCK, 2 * BLOCK):
                mask = ((np.arange(n) - off) % PERIOD) < BLOCK
                mix = {j: np.where(mask[:, None], h[j], c[j]) for j in ("shoulder", "hip", "knee")}
                plain = angle(mix["shoulder"], mix["hip"], mix["knee"]) - base
                fixed = angle(mix["shoulder"], mix["hip"], interp_knee(mix["knee"], mask)) - base
                for b, (lo, hi) in enumerate(BINS):
                    k = mask & (r["ref"] >= lo) & (r["ref"] < hi)
                    for v, a in (("plain", plain), ("fixed", fixed)):
                        acc.setdefault((v, b), {}).setdefault(s, []).append(a[k])
        for b, (lo, hi) in enumerate(BINS):
            cat = lambda v: {s: np.concatenate(x) for s, x in acc[(v, b)].items()}
            n = sum(len(x) for x in cat("plain").values())
            print(f"{('RTMPose' if m == 'rtm' else 'HMR2.0'):8s}{f'{lo}-{hi} deg':>16s}{n:8d}  {fmt(boot(cat('plain'), np.median)):24s}{fmt(boot(cat('fixed'), np.median)):24s}")
    print("\n  interpolation uses the true block positions (an upper bound).")


if __name__ == "__main__":
    rec = load_all()
    print(f"{len(rec)} subjects, {sum(len(r['ref']) for r in rec.values())} frames\n")
    test_a(rec)
    test_b(rec)
