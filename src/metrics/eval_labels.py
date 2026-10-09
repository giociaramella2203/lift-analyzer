"""Measure pose-model error against your hand labels.

Usage:
  python src/metrics/eval_labels.py <clip> [--lift deadlift] [--models folder1 folder2 ...] [--side left]
Defaults to the folders that exist among: <clip>, <clip>_rtm, <clip>_clean, <clip>_rtm_clean (under outputs/).
Reads labels/<clip>_*.csv (all sessions; with several sessions the mean click is used as the label).

For each model prints, on the labelled frames:
  shoulder / hip / knee error, in % of your labelled torso length (shoulder-hip distance), median and 90th percentile
  hip-angle error in degrees (median and 90th percentile)
  the same split by phase (ascent = plate crossing the knee, vs bottom) and by knee visible vs knee hidden
If you labelled in two sessions it also prints how much YOUR clicks differ between sessions: no model can be
judged more precisely than that.
Extra checks:
  --detail   prints the signed average offset (model minus your label, in px) per joint, and one line per frame.
             A consistent offset means a definition mismatch (where you click vs where the model puts the joint),
             not an occlusion failure.
  --debias   adds a second table with each model's average offset removed (leave-one-out), so only the random
             error is left. Use it when the signed offsets from --detail are large.
  --sheet    saves outputs/<clip>/label_check.png: each labelled frame, cropped, with your labels (yellow) and the
             two raw models (MediaPipe red, RTMPose blue). Needs data/raw/<clip>.mp4. Use it to check your own clicks.
Few labelled frames means wide uncertainty: treat differences of a few percent as noise.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lift_reps import LM, angle

J = ["shoulder", "hip", "knee"]


def load_labels(clip):
    files = sorted(Path("labels").glob(f"{clip}_*.csv"))
    if not files:
        raise SystemExit(f"No label files like labels/{clip}_*.csv")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df.status == "ok"].copy()
    if "knee_hidden" not in df:
        df["knee_hidden"] = 0
    df["knee_hidden"] = df["knee_hidden"].fillna(0).astype(int)
    return df


def detail(models, side, frames, S, hidden, torso):
    idx = [LM[side][j] for j in J]
    print("\n  Signed offset, model minus your label (px; x>0 = model further right, y>0 = further down), mean over frames")
    for m in models[:2]:
        p = Path("outputs") / m / "mediapipe_keypoints.npz"
        if not p.exists():
            continue
        kp = np.load(p)["keypoints"]
        print("  " + f"{m:24s}" + "  ".join(f"{j}: dx {np.nanmean(kp[frames, i, 0] - S[j][:, 0]):+6.1f} dy {np.nanmean(kp[frames, i, 1] - S[j][:, 1]):+6.1f}" for j, i in zip(J, idx)))
    print("\n  Per frame knee error, px (torso length in px): frame  hidden  " + "  ".join(models[:2]))
    kps = [np.load(Path("outputs") / m / "mediapipe_keypoints.npz")["keypoints"] for m in models[:2]]
    for k, f in enumerate(frames):
        e = [np.linalg.norm(kp[f, idx[2], :2] - S["knee"][k]) for kp in kps]
        print(f"    frame {f:5d}  {'hidden' if hidden[k] else 'visible'}  torso {torso[k]:5.0f}   " + "   ".join(f"{x:6.1f}" for x in e))


def sheet(clip, models, side, frames, S, hidden):
    cap = cv2.VideoCapture(str(Path("data/raw") / f"{clip}.mp4"))
    if not cap.isOpened():
        print("  (no data/raw video, skipping the sheet)")
        return
    idx = [LM[side][j] for j in J]
    kps = [np.load(Path("outputs") / m / "mediapipe_keypoints.npz")["keypoints"] for m in models[:2]]
    cols = [(0, 0, 255), (255, 120, 0)]
    tiles = []
    for k, f in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        pts = np.vstack([S[j][k] for j in J])
        x0, y0 = (pts.min(0) - 70).astype(int)
        x1, y1 = (pts.max(0) + 70).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        def draw(points, color, r):
            for a, b in zip(points[:-1], points[1:]):
                cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1, cv2.LINE_AA)
            for q in points:
                cv2.circle(img, tuple(int(v) for v in q), r, color, -1, cv2.LINE_AA)
        for kp, c in zip(kps, cols):
            draw([kp[f, i, :2] for i in idx], c, 3)
        draw([S[j][k] for j in J], (0, 255, 255), 4)
        t = cv2.resize(img[y0:y1, x0:x1], (240, int(240 * (y1 - y0) / max(x1 - x0, 1))))
        cv2.putText(t, f"{f} {'HIDDEN' if hidden[k] else ''}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    ncol = 4
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    sheet_img = np.vstack([np.hstack(tiles[i:i + ncol]) for i in range(0, len(tiles), ncol)])
    out = Path("outputs") / clip / "label_check.png"
    cv2.imwrite(str(out), sheet_img)
    print(f"  Saved {out}  (yellow = your labels, red = {models[0]}, blue = {models[1] if len(models) > 1 else '-'})")


def main(clip, lift, models, side, do_detail=False, do_sheet=False, do_debias=False):
    df = load_labels(clip)
    sessions = sorted(df.session.unique())
    cols = [f"{j}_{c}" for j in J for c in "xy"]
    gt = df.groupby("frame")[cols + ["phase", "knee_hidden"]].agg({**{c: "mean" for c in cols}, "phase": "first", "knee_hidden": "max"}).reset_index()
    print(f"{clip}: {len(gt)} labelled frame(s) from session(s) {', '.join(sessions)}")
    from itertools import combinations
    for sa, sb in combinations(sessions, 2):
        a, b = df[df.session == sa].set_index("frame"), df[df.session == sb].set_index("frame")
        common = a.index.intersection(b.index)
        if len(common):
            d = {j: np.hypot(a.loc[common, f"{j}_x"] - b.loc[common, f"{j}_x"], a.loc[common, f"{j}_y"] - b.loc[common, f"{j}_y"]) for j in J}
            torso = np.hypot(a.loc[common, "shoulder_x"] - a.loc[common, "hip_x"], a.loc[common, "shoulder_y"] - a.loc[common, "hip_y"])
            print("  Your own repeat error between sessions " + f"{sa}/{sb} ({len(common)} frames), median % of torso: "
                  + ", ".join(f"{j} {np.median(d[j] / torso) * 100:.1f}" for j in J))
    S = {j: gt[[f"{j}_x", f"{j}_y"]].to_numpy() for j in J}
    torso = np.linalg.norm(S["shoulder"] - S["hip"], axis=1)
    ang_gt = angle(S["shoulder"], S["hip"], S["knee"])
    frames = gt.frame.to_numpy()
    phase = gt.phase.to_numpy()
    hidden = gt.knee_hidden.to_numpy() == 1
    print(f"  {int(hidden.sum())} of these frames were marked knee-hidden (your estimate).")
    idx = [LM[side][j] for j in J]

    def table(title, debias):
        print(f"\n  {title}")
        print(f"  {'model':26s} {'subset':8s} {'shoulder':>12s} {'hip':>12s} {'knee':>12s} {'angle deg':>12s}")
        for m in models:
            p = Path("outputs") / m / "mediapipe_keypoints.npz"
            if not p.exists():
                continue
            kp = np.load(p)["keypoints"]
            P = {j: kp[frames, i, :2].copy() for j, i in zip(J, idx)}
            if debias and len(frames) >= 4:
                for j in J:       # leave-one-out: subtract the average offset measured on the OTHER frames
                    off = P[j] - S[j]
                    P[j] = P[j] - (np.nansum(off, axis=0) - off) / (len(frames) - 1)
            ang_p = angle(P["shoulder"], P["hip"], P["knee"])
            for name, mask in (("all", np.ones(len(frames), bool)), ("ascent", phase == "ascent"), ("bottom", phase == "bottom"), ("hid.knee", hidden), ("vis.knee", ~hidden)):
                if mask.sum() == 0:
                    continue
                cells = []
                for j in J:
                    e = np.linalg.norm(P[j] - S[j], axis=1)[mask] / torso[mask] * 100
                    cells.append(f"{np.nanmedian(e):4.1f} / {np.nanpercentile(e, 90):4.1f}")
                ea = np.abs(ang_p - ang_gt)[mask]
                cells.append(f"{np.nanmedian(ea):4.1f} / {np.nanpercentile(ea, 90):4.1f}")
                print(f"  {m:26s} {name:8s} " + " ".join(f"{c:>12s}" for c in cells) + f"   (n={int(mask.sum())})")

    table("error as % of torso length (median / 90th pct); hip angle error in degrees", False)
    if do_debias:
        table("SAME, after removing each model's average offset per joint (leave-one-out): what is left is the random error, "
              "not the difference in where you click vs where the model puts the joint", True)
    print("\nSmall samples: differences of a few percent are within noise. Behind the plate your own labels are uncertain.")
    if do_detail:
        detail(models, side, frames, S, hidden, torso)
    if do_sheet:
        sheet(clip, models, side, frames, S, hidden)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", default="deadlift")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--debias", action="store_true")
    a = ap.parse_args()
    models = a.models or [m for m in (a.clip, f"{a.clip}_rtm", f"{a.clip}_clean", f"{a.clip}_rtm_clean") if (Path("outputs") / m / "mediapipe_keypoints.npz").exists()]
    main(a.clip, a.lift, models, a.side, a.detail, a.sheet, a.debias)
