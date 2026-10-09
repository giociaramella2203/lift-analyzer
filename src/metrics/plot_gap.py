"""Figure for the README: hip-angle error on visible vs knee-hidden frames, for both models, plate clip and empty-bar clip.

Usage (from the repository root):
  python src/metrics/plot_gap.py            # writes docs/fig_gap.png
Each dot is one frame I labelled. Error = model hip angle (shoulder-hip-knee, 2D) minus my hand-labelled hip angle, degrees, signed;
labels averaged over sessions as in eval_gap.py. Bars are medians. The number above each pair is the gap, median hidden minus median
visible, with a 95% bootstrap interval over labelled frames (same procedure as eval_gap.py).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lift_reps import angle, LM                # noqa: E402
from eval_labels import load_labels, J         # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRAY, ACCENT, SURFACE = "#9a9994", "#2a78d6", "#fcfcfb"
L = LM["left"]
MODELS = [("RTMPose", "{clip}_rtm"), ("MediaPipe", "{clip}")]


def signed_errors(clip, folder):
    k = np.load(Path("outputs") / folder.format(clip=clip) / "mediapipe_keypoints.npz")["keypoints"]
    df = load_labels(clip)
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    g = df.groupby("frame")[cols + ["knee_hidden"]].agg({**{c: "mean" for c in cols}, "knee_hidden": "max"}).reset_index()
    g = g[g.frame < len(k)]
    fr = g.frame.to_numpy()
    S = {j: g[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    ang = angle(k[fr, L["shoulder"], :2], k[fr, L["hip"], :2], k[fr, L["knee"], :2])
    return ang - angle(S["shoulder"], S["hip"], S["knee"]), g.knee_hidden.to_numpy() == 1


def main(out="docs/fig_gap.png"):
    panels = [("deadlift_set2", "Plates (knee behind the plate)"), ("deadlift_set1", "Empty bar")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True, facecolor=SURFACE)
    jit = np.random.default_rng(1)
    for ax, (clip, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for m, (mname, folder) in enumerate(MODELS):
            e, hidden = signed_errors(clip, folder)
            x, v = e[hidden], e[~hidden]
            rng = np.random.default_rng(0)
            bs = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(v, len(v))) for _ in range(5000)]
            gap, lo, hi = np.median(x) - np.median(v), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
            base = m * 2.4
            for off, vals, col, lab in ((0, v, GRAY, "visible"), (1, x, ACCENT, "hidden")):
                pos = base + off
                ax.scatter(pos + jit.uniform(-0.13, 0.13, len(vals)), vals, s=40, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
                ax.hlines(np.median(vals), pos - 0.27, pos + 0.27, color=col if col == ACCENT else INK2, linewidth=2.2, zorder=4)
                ax.text(pos, -3.8, f"{lab}\nn={len(vals)}", ha="center", va="top", fontsize=8.5, color=INK2)
            ax.text(base + 0.5, 36.5, mname, ha="center", fontsize=9.5, color=INK, fontweight="bold")
            ax.text(base + 0.5, 33.6, f"gap {gap:+.1f}°", ha="center", fontsize=9.5, color=ACCENT if lo > 0 else INK2, fontweight="bold")
            ax.text(base + 0.5, 31.4, f"[{lo:+.1f}, {hi:+.1f}]", ha="center", fontsize=8.5, color=INK2)
        ax.set_xlim(-0.6, 4.2)
        ax.set_ylim(-4, 40)
        ax.set_xticks([])
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", colors=INK2, labelsize=9, length=0)
    axes[0].set_ylabel("hip-angle error, model minus my label (degrees)", fontsize=9.5, color=INK2)
    fig.text(0.01, 0.005, "Each dot = one labelled frame; bars = medians; gap = median hidden minus median visible, 95% bootstrap interval "
             "(blue when the interval excludes 0). The two clips were filmed differently.", fontsize=7.6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(out).parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
