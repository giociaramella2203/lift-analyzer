"""Look at what the tracker did: knee from the pose model (blue) vs knee from the tracker (green) over one rep.

Usage:  python src/metrics/show_track.py <clip> [--rep 2] [--n 8] [--lift deadlift] [--model-folder <clip>_rtm] [--track-folder <clip>_track]
Saves outputs/<track folder>/track_check_rep<N>.png : n frames spread over the rep, with frame number.
  blue dot  = pose model knee      green dot = tracker knee (hollow = tracker says "not visible")
  green line = tracker path from the start of the rep up to that frame
Needs data/raw/<clip>.mp4, outputs/<clip>/<lift>_reps.csv and the two keypoint folders.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pose"))
from track_knee import rep_segments, LM  # noqa: E402


def main(clip, lift, rep, n, side, model_folder, track_folder):
    video = Path("data/raw") / f"{clip}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mod = np.load(Path("outputs") / model_folder / "mediapipe_keypoints.npz")["keypoints"]
    trk = np.load(Path("outputs") / track_folder / "mediapipe_keypoints.npz")["keypoints"]
    segs = {r: (s, e) for r, s, e in rep_segments(Path("outputs") / clip / f"{lift}_reps.csv", fps, len(mod))}
    if rep not in segs:
        raise SystemExit(f"rep {rep} not found; available: {sorted(segs)}")
    s, e = segs[rep]
    k = LM[side]["knee"]
    frames = np.linspace(s, e, n).round().astype(int)
    scale = 0.45
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, None, fx=scale, fy=scale)
        path = (trk[s:f + 1, k, :2] * scale).astype(np.int32)
        if len(path) > 1:
            cv2.polylines(img, [path.reshape(-1, 1, 2)], False, (0, 220, 0), 1)
        m = tuple(int(v) for v in mod[f, k, :2] * scale)
        t = tuple(int(v) for v in trk[f, k, :2] * scale)
        cv2.circle(img, m, 6, (255, 80, 0), -1)
        cv2.circle(img, t, 6, (0, 220, 0), -1 if trk[f, k, 2] > 0.5 else 2)
        cv2.putText(img, str(f), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(img)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    out = Path("outputs") / track_folder / f"track_check_rep{rep}.png"
    cv2.imwrite(str(out), sheet)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--lift", default="deadlift")
    ap.add_argument("--rep", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--side", default="left")
    ap.add_argument("--model-folder", default=None)
    ap.add_argument("--track-folder", default=None)
    a = ap.parse_args()
    main(a.clip, a.lift, a.rep, a.n, a.side, a.model_folder or f"{a.clip}_rtm", a.track_folder or f"{a.clip}_track")
