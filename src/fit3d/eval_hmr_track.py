"""Does temporal context rescue HMR2.0 when a disc hides the knee for 1 s?  Evaluates the output of notebooks/hmr2_track_clips.ipynb.

Usage (from the repository root):
  python src/fit3d/eval_hmr_track.py            # reads outputs/fit3d_track/hmr2_track_keypoints.npz and outputs/fit3d_track/<subject>/meta.npz

Clips (src/fit3d/export_clips_tracker.py): 100 frames at 25 fps, a plate-sized disc on the true knee for the middle 25 frames, s04, s05,
s08, s10 (the subjects with bent-over frames).  HMR2.0 was run on every frame, clean and with the disc, with a box from the true joints.
For each subject and each temporal filter, inside the 25 disc frames: the median change of the 2D hip angle (shoulder-hip-knee, filtered disc
version minus the RAW per-frame clean version), and the same for the clean clip (the cost of the filter itself, e.g. lag).  Filters are applied
to the x and y of the three joints before the angle is computed (the interpolation only touches the knee):
  raw            no filter (the per-frame model, as in eval_hmr.py)
  median 5/13    causal running median over the last 5 / 13 frames (0.2 s / 0.5 s)
  interpolate    the KNEE is replaced inside the block by a straight line between the last frame before and the first frame after it; shoulder and
                 hip are kept as observed (an oracle: it knows where the block is; the same repair as before, now on contiguous frames)
Reading guide: a tracker that always trusts what it sees behaves like 'raw' or 'median'; only a predictor that distrusts the disc frames
can behave like 'interpolate'.  Four subjects: per-subject numbers are printed, no confidence interval is attempted.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import GT, angle  # noqa: E402

HMR = {"right": dict(shoulder=2, hip=9, knee=10), "left": dict(shoulder=5, hip=12, knee=13)}
D = Path("outputs/fit3d_track")


def causal_median(x, k):
    return np.array([np.median(x[max(0, i - k + 1): i + 1], axis=0) for i in range(len(x))])


def interpolate(x, block, joint):
    if joint != "knee":
        return x
    i0, i1 = np.where(block)[0][[0, -1]]
    a, b = max(i0 - 1, 0), min(i1 + 1, len(x) - 1)
    out = x.copy()
    for i in range(i0, i1 + 1):
        w = (i - a) / (b - a)
        out[i] = (1 - w) * x[a] + w * x[b]
    return out


FILTERS = {
    "raw": lambda x, block, joint: x,
    "median 5": lambda x, block, joint: causal_median(x, 5),
    "median 13": lambda x, block, joint: causal_median(x, 13),
    "interpolate knee (oracle)": interpolate,
}


def main():
    z = np.load(D / "hmr2_track_keypoints.npz")
    subjects = sorted({k.rsplit("_", 1)[0] for k in z.files})
    rows = []
    for s in subjects:
        meta = np.load(D / s / "meta.npz")
        block, gt = meta["hidden"].astype(bool), meta["gt2d"]
        gk = gt[:, GT[str(meta["side"])]["knee"]]
        torso = np.linalg.norm(gt[:, GT[str(meta["side"])]["shoulder"]] - gt[:, GT[str(meta["side"])]["hip"]], axis=1)
        c, h = z[f"{s}_clean"], z[f"{s}_hidden"]
        side = min(HMR, key=lambda q: np.median(np.linalg.norm(c[:, HMR[q]["knee"]] - gk, axis=1)))
        J = {k: v for k, v in HMR[side].items()}
        pts = lambda a: {j: a[:, i] for j, i in J.items()}
        ang = lambda p: angle(p["shoulder"], p["hip"], p["knee"])
        ref = ang(pts(c))
        print(f"{s}: {len(block)} frames, disc on {int(block.sum())}; clean hip angle inside the block, median {np.median(ref[block]):.0f} deg; "
              f"knee shift under the disc {np.median(np.linalg.norm(h[:, J['knee']] - c[:, J['knee']], axis=1)[block] / torso[block]):.2f} torso")
        r = {}
        for name, fn in FILTERS.items():
            for cond, a in (("disc", h), ("clean", c)):
                p = {j: fn(a[:, i], block, j) for j, i in J.items()}
                r[(name, cond)] = np.median((ang(p) - ref)[block])
        rows.append(r)
    print()
    print(f"{'filter':24s}" + "".join(f"{s:>14s}" for s in subjects) + f"{'median':>10s}   (first line per filter: disc; second: clean clip, the filter's own cost)")
    for name in FILTERS:
        for cond in ("disc", "clean"):
            v = [r[(name, cond)] for r in rows]
            print(f"{(name if cond == 'disc' else ''):24s}" + "".join(f"{x:+14.1f}" for x in v) + f"{np.median(v):+10.1f}")
    print("\nHip-angle change in degrees, median over the 25 disc frames; positive = the angle opens up.")


if __name__ == "__main__":
    main()
