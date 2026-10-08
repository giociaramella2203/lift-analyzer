"""Compare MediaPipe and RTMPose on the same clip, for one lift.

Usage:
  python src/metrics/compare_models.py <clip_name> --lift deadlift
Optional: --mp and --rtm choose other folders under outputs/ (e.g. the *_clean ones).
Needs outputs/<clip>/mediapipe_keypoints.npz (MediaPipe) and
      outputs/<clip>_rtm/mediapipe_keypoints.npz (RTMPose, from extract_rtmpose.py).

Prints, per model: best-visible side, frame-to-frame angle noise (same estimator as jitter.py),
number of reps found. Then, between the models: RMS difference of the smoothed angle and the
median distance between the same joint in the two models, as a % of image height.
IMPORTANT: agreement between two models is NOT accuracy. Both can be wrong in the same way.
Accuracy needs hand-labelled frames.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from lift_reps import LIFTS, LM, angle, fill_nans, smooth


def load(path, lift):
    d = np.load(path)
    kp = d["keypoints"]
    names = LIFTS[lift]["triple"]
    vis = {s: np.nanmean(kp[:, [LM[s][n] for n in names], 2]) for s in LM}
    side = max(vis, key=vis.get)
    pts = [kp[:, LM[side][n], :2] for n in names]
    raw = fill_nans(angle(*pts))
    d2 = raw[2:] - 2 * raw[1:-1] + raw[:-2]
    noise = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
    ang = smooth(angle(*pts))
    reps = len(find_peaks(-ang, prominence=LIFTS[lift]["min_drop"])[0])
    return dict(kp=kp, side=side, pts=pts, ang=ang, noise=noise, reps=reps, h=float(d["height"]))


def main(clip, lift, mp_dir=None, rtm_dir=None):
    base = Path("outputs")
    mp_dir = mp_dir or clip
    rtm_dir = rtm_dir or f"{clip}_rtm"
    a = load(base / mp_dir / "mediapipe_keypoints.npz", lift)
    b = load(base / rtm_dir / "mediapipe_keypoints.npz", lift)
    n = min(len(a["ang"]), len(b["ang"]))
    print(f"{clip} / {lift}   ({len(a['ang'])} vs {len(b['ang'])} frames, using {n})")
    print(f"  MediaPipe: side={a['side']:5s} noise={a['noise']:.2f} deg  reps={a['reps']}")
    print(f"  RTMPose  : side={b['side']:5s} noise={b['noise']:.2f} deg  reps={b['reps']}")
    rms = float(np.sqrt(np.mean((a["ang"][:n] - b["ang"][:n]) ** 2)))
    print(f"  RMS difference of the smoothed angle: {rms:.1f} deg")
    if a["side"] != b["side"]:
        print("  (the models picked different sides; the joint comparison below uses MediaPipe's side for both)")
    # Where do the models disagree? A big RMS can come from a few bad stretches.
    diff = np.abs(a["ang"][:n] - b["ang"][:n])
    fps = float(np.load(base / mp_dir / "mediapipe_keypoints.npz")["fps"])
    big = diff > 10
    print(f"  Median angle difference: {np.median(diff):.1f} deg; frames differing by more than 10 deg: {big.mean() * 100:.0f}%")
    segs, i = [], 0
    while i < n:
        if big[i]:
            j = i
            while j + 1 < n and big[j + 1]:
                j += 1
            if j - i + 1 >= 5:
                segs.append((i / fps, (j + 1) / fps))
            i = j + 1
        else:
            i += 1
    segs = sorted(segs, key=lambda t: t[0] - t[1])[:8]
    if segs:
        print("  Longest stretches with >10 deg disagreement (s): " + ", ".join(f"{x:.1f}-{y:.1f}" for x, y in sorted(segs)))
    names = LIFTS[lift]["triple"]
    for nm in names:
        pa = a["kp"][:n, LM[a["side"]][nm], :2]
        pb = b["kp"][:n, LM[a["side"]][nm], :2]
        dist = np.linalg.norm(pa - pb, axis=1)
        print(f"  {nm:9s} median distance between models: {np.nanmedian(dist) / a['h'] * 100:.1f}% of image height")
    print("Agreement is not accuracy: validate against hand-labelled frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", required=True, choices=list(LIFTS))
    ap.add_argument("--mp", default=None, help="MediaPipe folder under outputs/ (default: <clip>)")
    ap.add_argument("--rtm", default=None, help="RTMPose folder under outputs/ (default: <clip>_rtm)")
    a = ap.parse_args()
    main(a.clip, a.lift, a.mp, a.rtm)
