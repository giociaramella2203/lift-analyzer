"""Fit3D controlled occlusion experiment, step 2: evaluate the outputs of run_pose.py.

Usage (from the repository root):
  python src/fit3d/eval_occlusion.py                         # all finished files in outputs/fit3d/
  python src/fit3d/eval_occlusion.py --test                  # the 10-frame smoke-test file(s) instead
  python src/fit3d/eval_occlusion.py --train s03 s04 s05 --heldout s07 s08 s09 s10 s11

Design. Each frame exists in three versions: clean, hidden (opaque disc on the true knee), control (same disc, knee visible).
The ground truth is the projected 3D joint set, so nothing here depends on my guesses.
Hip angle = shoulder-hip-knee in 2D on the leg nearer the camera. Because the same model sees the same frame with and
without the disc, the model-versus-annotation offset cancels: delta = angle(version) - angle(clean).
All intervals are 95% bootstrap intervals over SUBJECTS (frames of one subject are not independent).

Parts
  1. effect of the disc on the hip angle and on the knee position (hidden vs control), both models
  2. does the model know? knee confidence and two label-free signals, with AUC (hidden vs clean+control)
  3. detection thresholds tuned on the training subjects, applied unchanged to the held-out subjects
  4. repair on synthetic sequences (disc over the knee in blocks of frames, clean elsewhere): oracle interpolation
     (uses the true block positions) vs detect-then-interpolate (threshold from part 3)
Caveat: a drawn disc is not a real plate (no shadow, no motion blur, no depth cue).
"""
import argparse
from pathlib import Path

import numpy as np

LM = {"left": dict(shoulder=11, hip=23, knee=25), "right": dict(shoulder=12, hip=24, knee=26)}
GT = {"A": dict(shoulder=11, hip=1, knee=2), "B": dict(shoulder=14, hip=4, knee=5)}
CONDS = ("clean", "hidden", "control")
MODELS = ("rtm", "mp")
NAMES = {"rtm": "RTMPose", "mp": "MediaPipe"}
BLOCK, PERIOD = 6, 18          # synthetic sequences: 6 hidden samples out of every 18


def angle(a, b, c):
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def load(path):
    z = np.load(path)
    subj, cam = Path(path).stem.replace("_test", "").split("_")[:2]
    g = GT[str(z["side"])]
    gt = z["gt2d"]
    d = dict(subj=subj, cam=cam, n=len(z["frames"]), gt_knee=gt[:, g["knee"]],
             torso=np.linalg.norm(gt[:, g["shoulder"]] - gt[:, g["hip"]], axis=1))
    for m in MODELS:
        # model side = the knee closest to the true knee on the clean frames
        dist = {s: np.nanmedian(np.linalg.norm(z[f"{m}_clean"][:, LM[s]["knee"], :2] - d["gt_knee"], axis=1)) for s in LM}
        s = min(dist, key=lambda k: np.nan_to_num(dist[k], nan=1e9))
        for c in CONDS:
            k = z[f"{m}_{c}"]
            d[f"{m}_{c}"] = {j: k[:, LM[s][j], :2] for j in ("shoulder", "hip", "knee")}
            d[f"{m}_{c}"]["conf"] = np.nan_to_num(k[:, LM[s]["knee"], 2], nan=0.0)
    return d


def hip_angle(p):
    return angle(p["shoulder"], p["hip"], p["knee"])


def boot(by_subj, stat, reps=2000, seed=0):
    """stat(concatenated values) with a bootstrap over subjects. by_subj: dict subj -> array."""
    ids = list(by_subj)
    allv = np.concatenate([by_subj[i] for i in ids])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        v = np.concatenate([by_subj[i] for i in rng.choice(ids, len(ids))])
        v = v[~np.isnan(v)]
        if len(v):
            bs.append(stat(v))
    allv = allv[~np.isnan(allv)]
    return stat(allv), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fmt(t, unit="", dec=1):
    return f"{t[0]:+.{dec}f}{unit} [{t[1]:+.{dec}f}, {t[2]:+.{dec}f}]"


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def by_subject(files, fn):
    out = {}
    for f in files:
        out.setdefault(f["subj"], []).append(fn(f))
    return {k: np.concatenate(v) for k, v in out.items()}


def signals(f, c):
    """Label-free signals, larger = more likely hidden."""
    r, m = f[f"rtm_{c}"], f[f"mp_{c}"]
    return {"rtm_conf": 1 - r["conf"], "mp_conf": 1 - m["conf"],
            "disagree": np.linalg.norm(r["knee"] - m["knee"], axis=1) / f["torso"]}


def tune(x, y):
    ok = ~np.isnan(x)
    best, thr = -9, None
    for t in np.unique(x[ok]):
        fl = ok & (x >= t)
        j = fl[y].mean() - fl[~y].mean()
        if j > best:
            best, thr = j, t
    return thr, best


def interp_knee(knee, bad):
    out = knee.copy()
    t = np.arange(len(knee))
    good = ~bad & ~np.isnan(knee[:, 0])
    if good.sum() >= 2 and bad.any():
        for c in (0, 1):
            out[bad, c] = np.interp(t[bad], t[good], knee[good, c])
    return out


def main(files, train, heldout):
    F = [load(p) for p in files]
    print(f"{len(F)} files, subjects: {sorted({f['subj'] for f in F})}, {sum(f['n'] for f in F)} frames per version\n")

    print("1. Effect of the disc on the hip angle and the knee position (relative to the same frame without the disc)")
    print(f"{'model':10s}{'version':9s}{'miss %':>7s}  {'hip-angle change, deg (median)':34s}{'|change| > 10 deg':>18s}  {'knee shift / torso':>18s}")
    for m in MODELS:
        for c in ("hidden", "control"):
            def dang(f, m=m, c=c):
                return hip_angle(f[f"{m}_{c}"]) - hip_angle(f[f"{m}_clean"])

            def dknee(f, m=m, c=c):
                return np.linalg.norm(f[f"{m}_{c}"]["knee"] - f[f"{m}_clean"]["knee"], axis=1) / f["torso"]
            da, dk = by_subject(F, dang), by_subject(F, dknee)
            allv = np.concatenate(list(da.values()))
            miss = 100 * np.isnan(allv).mean()
            big = boot(da, lambda v: 100 * (np.abs(v) > 10).mean())
            kk = boot(dk, np.median)
            print(f"{NAMES[m]:10s}{c:9s}{miss:7.0f}  {fmt(boot(da, np.median)):34s}{big[0]:12.0f}% [{big[1]:.0f},{big[2]:.0f}]  {kk[0]:6.2f} [{kk[1]:.2f}, {kk[2]:.2f}]")
    print("   hidden minus control is the effect of hiding the knee; control tells how much a disc elsewhere in the picture moves the model.")
    print("   'miss' = frames where the model gave no pose at all (excluded from the angle columns).\n")

    print("1b. The same, split by posture (hip angle of the model on the CLEAN frame: small = bent over, large = upright)")
    print("    Exploratory: the bins were chosen after seeing that the pooled median hides a heavy tail.")
    print(f"{'model':10s}{'clean hip angle':>16s}{'frames':>8s}   {'hidden: change in hip angle, deg':34s}{'|change| > 10 deg':>18s}")
    for m in MODELS:
        for lo, hi in ((0, 100), (100, 140), (140, 181)):
            def dang_bin(f, m=m, lo=lo, hi=hi):
                ref = hip_angle(f[f"{m}_clean"])
                d = hip_angle(f[f"{m}_hidden"]) - ref
                return np.where((ref >= lo) & (ref < hi), d, np.nan)
            da = by_subject(F, dang_bin)
            n = int(sum((~np.isnan(v)).sum() for v in da.values()))
            big = boot(da, lambda v: 100 * (np.abs(v) > 10).mean())
            print(f"{NAMES[m]:10s}{f'{lo}-{hi} deg':>16s}{n:8d}   {fmt(boot(da, np.median)):34s}{big[0]:12.0f}% [{big[1]:.0f},{big[2]:.0f}]")
    print("    The disc matters mainly when the lifter is bent over; upright frames are barely affected.\n")

    print("2. Does the model know? Knee confidence and label-free signals (AUC: hidden frames vs clean+control frames)")
    print(f"{'signal':12s}{'median clean':>13s}{'hidden':>9s}{'control':>9s}   {'AUC hidden vs rest':28s}{'AUC hidden vs control':>24s}")
    for name in ("rtm_conf", "mp_conf", "disagree"):
        med = {c: np.nanmedian(np.concatenate([signals(f, c)[name] for f in F])) for c in CONDS}
        ids = sorted({f["subj"] for f in F})
        rng = np.random.default_rng(0)

        def aucs(sub):
            ff = [f for f in F if f["subj"] in sub]
            h = np.concatenate([signals(f, "hidden")[name] for f in ff])
            cl = np.concatenate([signals(f, "clean")[name] for f in ff])
            co = np.concatenate([signals(f, "control")[name] for f in ff])
            return auc(h, np.concatenate([cl, co])), auc(h, co)
        a = aucs(ids)
        bs = np.array([aucs(list(rng.choice(ids, len(ids)))) for _ in range(300)])
        ci = [np.nanpercentile(bs[:, i], [2.5, 97.5]) for i in (0, 1)]
        print(f"{name:12s}{med['clean']:13.2f}{med['hidden']:9.2f}{med['control']:9.2f}   {a[0]:.2f} [{ci[0][0]:.2f}, {ci[0][1]:.2f}]{'':8s}{a[1]:.2f} [{ci[1][0]:.2f}, {ci[1][1]:.2f}]")
    print("   rtm_conf / mp_conf = 1 - knee score (MediaPipe: no pose = 1); disagree = RTMPose-MediaPipe knee distance / torso.\n")

    trn = [f for f in F if f["subj"] in train]
    tst = [f for f in F if f["subj"] in heldout]
    if not trn or not tst:
        print("3-4 skipped: need both training and held-out subjects among the files.")
        return
    print(f"3. Detection thresholds tuned on {sorted(train)} (Youden J), applied unchanged to {sorted(heldout)}")
    thr = {}
    print(f"{'signal':12s}{'threshold':>10s}{'train TPR-FPR':>15s}   held-out flagged: hidden / clean / control")
    for name in ("rtm_conf", "mp_conf", "disagree"):
        x = np.concatenate([signals(f, c)[name] for f in trn for c in ("hidden", "clean", "control")])
        y = np.concatenate([np.full(f["n"], c == "hidden") for f in trn for c in ("hidden", "clean", "control")])
        thr[name], j = tune(x, y)
        rate = {c: np.mean(np.concatenate([np.nan_to_num(signals(f, c)[name], nan=-1e9) >= thr[name] for f in tst])) for c in CONDS}
        print(f"{name:12s}{thr[name]:10.2f}{j:15.2f}   {rate['hidden'] * 100:5.0f}% / {rate['clean'] * 100:3.0f}% / {rate['control'] * 100:3.0f}%")
    print()

    print(f"4. Repair on synthetic sequences ({BLOCK} hidden samples in every {PERIOD}), RTMPose knee, held-out subjects")
    print("   hip-angle change vs the same sequence without any disc (median deg, 95% interval over subjects)")
    print(f"{'variant':28s}{'inside hidden blocks':>30s}{'outside blocks (damage)':>30s}")

    def seq(f):
        mask = (np.arange(f["n"]) % PERIOD) < BLOCK
        mix = {k: np.where(mask[:, None], f["rtm_hidden"][k], f["rtm_clean"][k]) for k in ("shoulder", "hip", "knee")}
        mix["conf"] = np.where(mask, f["rtm_hidden"]["conf"], f["rtm_clean"]["conf"])
        mpm = {k: np.where(mask[:, None], f["mp_hidden"][k], f["mp_clean"][k]) for k in ("knee",)}
        mpm["conf"] = np.where(mask, f["mp_hidden"]["conf"], f["mp_clean"]["conf"])
        sig = {"rtm_conf": 1 - mix["conf"], "mp_conf": 1 - mpm["conf"],
               "disagree": np.linalg.norm(mix["knee"] - mpm["knee"], axis=1) / f["torso"]}
        return mask, mix, sig

    variants = ["no repair", "oracle interpolation"] + [f"detect ({n}) + interp" for n in thr]
    res = {v: ({}, {}) for v in variants}
    for f in tst:
        mask, mix, sig = seq(f)
        ref = hip_angle(f["rtm_clean"])
        for v in variants:
            if v == "no repair":
                knee = mix["knee"]
            elif v == "oracle interpolation":
                knee = interp_knee(mix["knee"], mask)
            else:
                name = v.split("(")[1].split(")")[0]
                flag = np.nan_to_num(sig[name], nan=-1e9) >= thr[name]
                knee = interp_knee(mix["knee"], flag)
            d = angle(mix["shoulder"], mix["hip"], knee) - ref
            res[v][0].setdefault(f["subj"], []).append(d[mask])
            res[v][1].setdefault(f["subj"], []).append(d[~mask])
    for v in variants:
        ins = {k: np.concatenate(x) for k, x in res[v][0].items()}
        out = {k: np.concatenate(x) for k, x in res[v][1].items()}
        print(f"{v:28s}{fmt(boot(ins, np.median)):>30s}{fmt(boot(out, np.median)):>30s}")
    print("   oracle interpolation uses the true block positions (upper bound); detect+interp does not.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/fit3d")
    ap.add_argument("--test", action="store_true", help="use the *_test.npz smoke-test files")
    ap.add_argument("--train", nargs="*", default=["s03", "s04", "s05"])
    ap.add_argument("--heldout", nargs="*", default=["s07", "s08", "s09", "s10", "s11"])
    a = ap.parse_args()
    paths = sorted(p for p in Path(a.dir).glob("s*_*.npz") if p.stem.endswith("_test") == a.test)
    if not paths:
        raise SystemExit("no files found")
    main(paths, a.train, a.heldout)
