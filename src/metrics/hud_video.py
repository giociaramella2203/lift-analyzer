"""Burn a rep-by-rep HUD onto the annotated video.

Usage:
  python src/metrics/hud_video.py <clip_name> --lift deadlift [--ignore 1] [--rom-tol 15] [--tempo-tol 0.35]
Needs, in outputs/<clip_name>/:
  mediapipe_annotated.mp4   (from extract_mediapipe.py)
  mediapipe_keypoints.npz   (from extract_mediapipe.py)
  <lift>_reps.csv           (from lift_reps.py)
Writes outputs/<clip_name>/<lift>_hud.mp4 and <lift>_reps_eval.csv.

What it shows (everything is computed from the angle curve):
  Reps k/N    reps completed so far / total detected
  Clean       reps that are complete AND within --rom-tol degrees of the set's median
              bottom angle AND within --tempo-tol (fraction) of the median ascent time
  Slowdown    ascent time of the latest rep vs the median of the first 3 valid reps
  one-liner   why the latest rep is clean or not
"Clean" means consistent with the rest of this set. It is NOT a verdict on
technique or safety, and the slowdown is a proxy limited by the frame rate.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lift_reps import LIFTS, LM, angle, smooth   # same folder as this script

GREEN, ORANGE, GRAY, WHITE = (80, 200, 80), (0, 165, 255), (190, 190, 190), (255, 255, 255)


def evaluate(df, ignore, rom_tol, tempo_tol):
    valid = df[df.complete & ~df.rep.isin(ignore)]
    med_min = valid.min_deg.median() if len(valid) else np.nan
    med_asc = valid.ascent_s.median() if len(valid) else np.nan
    base = valid.ascent_s.iloc[:3].median() if len(valid) else np.nan
    out = []
    for r in df.itertuples():
        d_rom = r.min_deg - med_min
        d_asc = r.ascent_s / med_asc - 1
        slow = r.ascent_s / base - 1
        if r.rep in ignore:
            status, text = "EXCL", f"Rep {r.rep}: excluded (setup, not a work rep)"
        elif not r.complete:
            status, text = "CUT", f"Rep {r.rep}: clip cut it short, timing not reliable"
        else:
            issues = []
            if abs(d_rom) > rom_tol:
                issues.append(f"{abs(d_rom):.0f} deg {'deeper' if d_rom < 0 else 'shallower'} than median")
            if abs(d_asc) > tempo_tol:
                issues.append(f"ascent {d_asc * 100:+.0f}% vs median")
            if issues:
                status, text = "CHECK", f"Rep {r.rep}: " + "; ".join(issues)
            else:
                status = "CLEAN"
                text = f"Rep {r.rep}: clean, bottom {d_rom:+.0f} deg and ascent {d_asc * 100:+.0f}% vs median"
        out.append(dict(rep=r.rep, status=status, text=text, d_rom_deg=round(float(d_rom), 1),
                        d_ascent_pct=round(float(d_asc) * 100, 1), slowdown_pct=round(float(slow) * 100, 1)))
    return pd.DataFrame(out)


def put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap(text, scale, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def shade(img, y0, y1):
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)


def main(clip, lift, ignore, rom_tol, tempo_tol, hud_bottom=False):
    folder = Path("outputs") / clip
    cfg = LIFTS[lift]
    df = pd.read_csv(folder / f"{lift}_reps.csv")
    d = np.load(folder / "mediapipe_keypoints.npz")
    kp, fps = d["keypoints"], float(d["fps"])
    ev = evaluate(df, set(ignore), rom_tol, tempo_tol)
    ev.to_csv(folder / f"{lift}_reps_eval.csv", index=False)

    # live angle + clip-level confidence (same side selection as lift_reps.py)
    names = cfg["triple"]
    vis = {s: np.nanmean(kp[:, [LM[s][n] for n in names], 2]) for s in LM}
    side = max(vis, key=vis.get)
    ang = smooth(angle(*[kp[:, LM[side][n], :2] for n in names]))
    v = kp[:, [LM[side][n] for n in names], 2]
    low_pct = float(np.mean(~(np.min(v, axis=1) >= 0.5)) * 100)   # undetected frames count as low

    cap = cv2.VideoCapture(str(folder / "mediapipe_annotated.mp4"))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(folder / f"{lift}_hud.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    ends = [min(int(round((r.liftoff_s + r.ascent_s) * fps)), n_frames - 1) for r in df.itertuples()]

    s = max(0.45, w / 800)
    lh = int(34 * s)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        done = [k for k, e in enumerate(ends) if e <= i]
        k_last = done[-1] if done else None
        n_clean = int((ev.status.iloc[done] == "CLEAN").sum()) if done else 0
        counted = [k for k in done if ev.status.iloc[k] in ("CLEAN", "CHECK")]
        slow = ev.slowdown_pct.iloc[counted[-1]] if counted else None

        oy = h - int(lh * 4.4) if hud_bottom else 0      # panel offset (bottom option)
        shade(frame, oy, oy + int(lh * 4.4))
        put(frame, f"{lift.upper()}   Reps {len(done)}/{len(df)}   Clean {n_clean}", (10, oy + lh), s, WHITE, 2)
        put(frame, f"Tempo vs first 3 reps: " + (f"{slow:+.0f}%" if slow is not None else "n/a"),
            (10, oy + lh * 2), s * 0.85, WHITE)
        put(frame, f"{cfg['label']}: {ang[min(i, len(ang) - 1)]:.0f} deg", (10, oy + lh * 3), s * 0.85, WHITE)
        put(frame, f"Pose confidence {vis[side]:.2f} | low-conf frames {low_pct:.0f}%",
            (10, oy + int(lh * 3.9)), s * 0.75, GRAY)

        if k_last is not None:
            st = ev.status.iloc[k_last]
            color = GREEN if st == "CLEAN" else ORANGE if st == "CHECK" else GRAY
            lines = wrap(ev.text.iloc[k_last], s * 0.8, w - 20)
            ly = 0 if hud_bottom else h - int(lh * (len(lines) + 1))   # one-liner goes opposite the panel
            shade(frame, ly, ly + int(lh * (len(lines) + 1)))
            for j, line in enumerate(lines):
                put(frame, line, (10, ly + int(lh * (j + 1.0))), s * 0.8, color, 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    print(ev[["rep", "status", "d_rom_deg", "d_ascent_pct", "slowdown_pct"]].to_string(index=False))
    print(f"Clean reps: {int((ev.status == 'CLEAN').sum())} of {len(ev)} detected.")
    print(f"Saved {lift}_hud.mp4 and {lift}_reps_eval.csv to {folder}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="clip name, e.g. deadlift_set1 (folder under outputs/)")
    ap.add_argument("--lift", required=True, choices=list(LIFTS))
    ap.add_argument("--ignore", type=int, nargs="*", default=[], help="rep numbers to exclude (e.g. setup rep)")
    ap.add_argument("--rom-tol", type=float, default=15.0, help="degrees from the median bottom angle")
    ap.add_argument("--tempo-tol", type=float, default=0.35, help="fraction from the median ascent time")
    ap.add_argument("--hud-bottom", action="store_true", help="put the stats panel at the bottom (use when the action is at the top of the frame)")
    a = ap.parse_args()
    main(a.clip, a.lift, a.ignore, a.rom_tol, a.tempo_tol, a.hud_bottom)
