"""Does the disc change the person box that the detector hands to RTMPose?  Step 2: evaluate the outputs of run_box.py.

Usage (from the repository root):
  python src/fit3d/eval_box.py

Per posture bin (hip angle of RTMPose on the clean frame, same bins as before):
  box IoU / area   overlap between the detector's person box on the clean image and on the disc image (1 = identical) and the
                   ratio of box areas (disc / clean); 'changed' = share of frames with IoU below 0.9
  free box         hip-angle change caused by the disc when the detector runs on each image (the original experiment)
  fixed box        the same, when the pose model is always given the CLEAN box
  shifts           median shift of the model's hip, shoulder and knee (torso lengths), free box vs fixed box
Reading guide
  * box changes a lot and the fixed-box change is much smaller than the free-box change -> the box explains a good part of the effect
  * box barely changes, or fixed box ~ free box                                         -> the box is not the reason
  * fixed box change still large, knee shift still large                                -> the disc really disturbs the pose model's reading
Also printed: a sanity check that the clean frame gives the same keypoints with the fixed box as in the original run.
Intervals are 95% bootstrap intervals over subjects.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import LM, angle, boot, by_subject, fmt, load  # noqa: E402

BINS = ((0, 100), (100, 140), (140, 181))
J = ("shoulder", "hip", "knee")


def iou(a, b):
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a) + area(b) - inter), area(b) / area(a)


def main():
    F = []
    for p in sorted(Path("outputs/fit3d").glob("s*_*.npz")):
        if p.stem.endswith("_test"):
            continue
        bp = p.with_name(f"box_{p.stem}.npz")
        if not bp.exists():
            print("missing", bp.name)
            continue
        f, z, b = load(p), np.load(p), np.load(bp)
        assert len(b["frames"]) == f["n"], p.name
        dist = {s: np.nanmedian(np.linalg.norm(z["rtm_clean"][:, LM[s]["knee"], :2] - f["gt_knee"], axis=1)) for s in LM}
        s = min(dist, key=lambda k: np.nan_to_num(dist[k], nan=1e9))
        for c in ("clean", "hidden"):
            f[f"fix_{c}"] = {j: b[f"fixed_{c}"][:, LM[s][j], :2] for j in J}
        f["iou"], f["area"] = iou(b["box_clean"], b["box_hidden"])
        F.append(f)
    if not F:
        raise SystemExit("no box files; run run_box.py first")
    print(f"{len(F)} files, {sum(f['n'] for f in F)} frames per version\n")

    ang = lambda d: angle(d["shoulder"], d["hip"], d["knee"])
    sd = np.concatenate([np.linalg.norm(f["fix_clean"]["knee"] - f["rtm_clean"]["knee"], axis=1) / f["torso"] for f in F])
    print(f"sanity: knee on the clean frame, fixed box vs original run: median difference {np.nanmedian(sd):.3f} torso, 95th percentile {np.nanpercentile(sd, 95):.3f}\n")

    print(f"{'clean hip angle':>16s}{'frames':>8s}  {'box IoU':>8s}{'changed':>9s}{'area ratio':>11s}  {'free box: hip-angle change':28s}{'fixed box: hip-angle change':28s}")
    shifts = []
    for lo, hi in BINS:
        def pick(fn, lo=lo, hi=hi):
            return by_subject(F, lambda f: np.where((ang(f["rtm_clean"]) >= lo) & (ang(f["rtm_clean"]) < hi), fn(f), np.nan))
        n = int(sum(np.sum((ang(f["rtm_clean"]) >= lo) & (ang(f["rtm_clean"]) < hi)) for f in F))
        io = boot(pick(lambda f: f["iou"]), np.median)
        ch = boot(pick(lambda f: (f["iou"] < 0.9).astype(float)), np.mean)
        ar = boot(pick(lambda f: f["area"]), np.median)
        free = boot(pick(lambda f: ang(f["rtm_hidden"]) - ang(f["rtm_clean"])), np.median)
        fixd = boot(pick(lambda f: ang(f["fix_hidden"]) - ang(f["fix_clean"])), np.median)
        print(f"{f'{lo}-{hi} deg':>16s}{n:8d}  {io[0]:8.2f}{100 * ch[0]:8.0f}%{ar[0]:11.2f}  {fmt(free):28s}{fmt(fixd):28s}")
        row = []
        for j in J:
            fr = boot(pick(lambda f, j=j: np.linalg.norm(f["rtm_hidden"][j] - f["rtm_clean"][j], axis=1) / f["torso"]), np.median)[0]
            fx = boot(pick(lambda f, j=j: np.linalg.norm(f["fix_hidden"][j] - f["fix_clean"][j], axis=1) / f["torso"]), np.median)[0]
            row.append(f"{j} {fr:.2f} -> {fx:.2f}")
        shifts.append((f"{lo}-{hi} deg", row))
    print("\nShift of the model's joints caused by the disc, torso lengths, free box -> fixed box (medians)")
    for name, row in shifts:
        print(f"{name:>16s}   " + "     ".join(row))


if __name__ == "__main__":
    main()
