"""Repair and detection split by posture (follow-up to eval_occlusion.py part 3-4, which pooled all frames).  No model is re-run.

Usage (from the repository root):
  python src/fit3d/eval_repair_posture.py

Same synthetic sequences as eval_occlusion.py (the disc version for blocks of 6 sampled frames out of every 18, the clean version
elsewhere), but here the blocks are placed at three offsets (0, 6, 12) so that EVERY frame is hidden exactly once; frames are then
grouped by posture (hip angle of RTMPose on the clean frame: under 100, 100-140, 140 and above).
Thresholds are tuned on the training subjects (s03 s04 s05, Youden J, pooled over postures) and applied unchanged to the
held-out subjects (s07-s11), as before.
Variants (RTMPose knee):  no repair | oracle interpolation (true block positions, upper bound) | detect (signal) + interpolation
Tables
  A  detector: share of frames flagged inside the hidden blocks (hits) and outside them (false alarms), per posture
  B  hip-angle change vs the same sequence without any disc, per posture:
       inside blocks = what a repair is for;  outside blocks = damage done by flagging frames that were fine.
     Median in degrees and share of frames changing by more than 10 degrees; 95% intervals over subjects.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import (BLOCK, PERIOD, angle, boot, fmt, hip_angle, interp_knee, load, signals, tune)  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
SIGNALS = ("rtm_conf", "mp_conf", "disagree")


def seq(f, off):
    n = f["n"]
    mask = ((np.arange(n) - off) % PERIOD) < BLOCK
    pick = lambda m, k: np.where(mask[:, None], f[f"{m}_hidden"][k], f[f"{m}_clean"][k])
    mix = {k: pick("rtm", k) for k in ("shoulder", "hip", "knee")}
    conf = np.where(mask, f["rtm_hidden"]["conf"], f["rtm_clean"]["conf"])
    mpk = pick("mp", "knee")
    mpc = np.where(mask, f["mp_hidden"]["conf"], f["mp_clean"]["conf"])
    sig = {"rtm_conf": 1 - conf, "mp_conf": 1 - mpc, "disagree": np.linalg.norm(mix["knee"] - mpk, axis=1) / f["torso"]}
    return mask, mix, sig


def main(train, heldout):
    paths = sorted(p for p in Path("outputs/fit3d").glob("s*_*.npz") if not p.stem.endswith("_test"))
    F = [load(p) for p in paths]
    trn = [f for f in F if f["subj"] in train]
    tst = [f for f in F if f["subj"] in heldout]
    if not trn or not tst:
        raise SystemExit("need training and held-out subjects")
    thr = {}
    for name in SIGNALS:
        x = np.concatenate([signals(f, c)[name] for f in trn for c in ("hidden", "clean", "control")])
        y = np.concatenate([np.full(f["n"], c == "hidden") for f in trn for c in ("hidden", "clean", "control")])
        thr[name], _ = tune(x, y)
    print(f"thresholds tuned on {sorted(train)}: " + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()))
    print(f"held-out subjects: {sorted(heldout)}; every frame is hidden once (3 block offsets)\n")

    variants = ["no repair", "oracle interpolation"] + [f"detect ({n}) + interp" for n in SIGNALS]
    acc, det = {}, {}
    for f in tst:
        ref = hip_angle(f["rtm_clean"])
        binid = np.full(f["n"], -1)
        for b, (lo, hi) in enumerate(BINS):
            binid[(ref >= lo) & (ref < hi)] = b
        for off in (0, BLOCK, 2 * BLOCK):
            mask, mix, sig = seq(f, off)
            flags = {n: np.nan_to_num(sig[n], nan=-1e9) >= thr[n] for n in SIGNALS}
            for v in variants:
                if v == "no repair":
                    knee = mix["knee"]
                elif v == "oracle interpolation":
                    knee = interp_knee(mix["knee"], mask)
                else:
                    knee = interp_knee(mix["knee"], flags[v.split("(")[1].split(")")[0]])
                d = angle(mix["shoulder"], mix["hip"], knee) - ref
                for b in range(len(BINS)):
                    for io, sel in (("in", mask), ("out", ~mask)):
                        k = sel & (binid == b)
                        acc.setdefault((v, b, io), {}).setdefault(f["subj"], []).append(d[k])
            for n in SIGNALS:
                for b in range(len(BINS)):
                    for io, sel in (("in", mask), ("out", ~mask)):
                        k = sel & (binid == b)
                        det.setdefault((n, b, io), {}).setdefault(f["subj"], []).append(flags[n][k].astype(float))

    cat = lambda d: {s: np.concatenate(v) for s, v in d.items()}
    print("A. Detector: share of frames flagged, by posture (hits = inside hidden blocks, false alarms = outside)")
    print(f"{'signal':12s}{'clean hip angle':>16s}{'hits':>22s}{'false alarms':>22s}")
    for n in SIGNALS:
        for b, (lo, hi) in enumerate(BINS):
            h, fa = boot(cat(det[(n, b, "in")]), lambda v: 100 * v.mean()), boot(cat(det[(n, b, "out")]), lambda v: 100 * v.mean())
            print(f"{n:12s}{f'{lo}-{hi} deg':>16s}{h[0]:12.0f}% [{h[1]:.0f}, {h[2]:.0f}]{fa[0]:12.0f}% [{fa[1]:.0f}, {fa[2]:.0f}]")
    print()

    print("B. Hip-angle change vs the same sequence without any disc (median deg [95% interval]; share of frames above 10 deg)")
    for b, (lo, hi) in enumerate(BINS):
        nin = int(sum(len(a) for s in acc[("no repair", b, "in")].values() for a in s))
        print(f"\n  clean hip angle {lo}-{hi} deg   ({nin} frames inside blocks)")
        print(f"  {'variant':28s}{'inside blocks':24s}{'>10 deg':>9s}   {'outside blocks (damage)':24s}{'>10 deg':>9s}")
        for v in variants:
            row = []
            for io in ("in", "out"):
                d = cat(acc[(v, b, io)])
                m = boot(d, np.median)
                big = boot(d, lambda x: 100 * (np.abs(x) > 10).mean())
                row.append((fmt(m), f"{big[0]:.0f}%"))
            print(f"  {v:28s}{row[0][0]:24s}{row[0][1]:>9s}   {row[1][0]:24s}{row[1][1]:>9s}")
    print("\n  oracle interpolation uses the true block positions (upper bound); detect+interp does not.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="*", default=["s03", "s04", "s05"])
    ap.add_argument("--heldout", nargs="*", default=["s07", "s08", "s09", "s10", "s11"])
    a = ap.parse_args()
    main(a.train, a.heldout)
