"""Export short contiguous clips (clean and disc-occluded) for a Colab run of the 4D-Humans video tracker (PHALP).

Usage (from the repository root; needs only opencv and numpy):
  python src/fit3d/export_clips_tracker.py                       # s04, s05, s08, s10, camera 60457274
  python src/fit3d/export_clips_tracker.py --subjects s08        # one subject, to test
Writes outputs/fit3d_track/<subject>/{clean,hidden}.mp4, outputs/fit3d_track/<subject>/meta.npz and outputs/fit3d_track.zip.
Files stay in outputs/ (git-ignored): Fit3D must not be redistributed, so upload the zip only to your own Colab session.

Why clips: the single-image experiments used every 12th video frame, but a tracker needs consecutive frames.
Per subject, one window of 200 video frames (4 s at 50 fps), written at 25 fps (every 2nd frame, 100 frames):
  1.5 s clean context, then 1 s with the plate-sized disc on the TRUE projected knee (the same disc as before, drawn on every
  frame of the block), then 1.5 s clean.  The 'clean' clip is the same window without the disc, the reference.
The block is placed where RTMPose's clean hip angle is bent over (under 100 deg) for most of the saved samples, because that is where
the single-image models moved: the question is whether the tracker's temporal state resists the disc where they did not.
Subjects: the ones with bent-over frames (RTMPose hip angle under 100 deg on the clean frame: s04 50, s05 33, s08 16, s10 23 samples;
s09 has 3, s03, s07 and s11 none).  The tracker has no threshold to tune, so s04 and s05 can be used as well.
The window is chosen from RTMPose outputs only (not from any result with the disc), so it does not favour either outcome.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_occlusion import hip_angle, load  # noqa: E402
from run_pose import OUT, PLATE_RADIUS_M, ROOT, SIDES, draw_disc, load_cam, project  # noqa: E402

PRE, BLOCK, POST, STRIDE, FPS = 75, 50, 75, 2, 25      # video frames at 50 fps


def pick_block(subj, cam):
    """Start of the 50-frame block with the most bent-over samples, such that the 200-frame window fits in the saved range."""
    f, z = load(OUT / f"{subj}_{cam}.npz"), np.load(OUT / f"{subj}_{cam}.npz")
    fr, ref = z["frames"], hip_angle(f["rtm_clean"])
    best = None
    for b in range(int(fr.min()) + PRE, int(fr.max()) - POST - BLOCK + 1, 6):
        k = (fr >= b) & (fr < b + BLOCK)
        score = np.nansum(ref[k] < 100) + 0.01 * np.nansum(ref[k] < 140)
        if best is None or score > best[0]:
            best = (score, b, int(np.nansum(ref[k] < 100)), int(k.sum()))
    return best


def main(subjects, cam):
    dst = Path("outputs/fit3d_track")
    for subj in subjects:
        score, b, nbent, nsamp = pick_block(subj, cam)
        start, end = b - PRE, b + BLOCK + POST
        J = np.array(json.load(open(ROOT / subj / "joints3d_25" / "deadlift.json"))["joints3d_25"], float)
        R, T, K, dist = load_cam(subj, cam)
        zA = ((J[:, SIDES["A"]["hip"]] - T) @ R.T)[:, 2].mean()
        zB = ((J[:, SIDES["B"]["hip"]] - T) @ R.T)[:, 2].mean()
        side = "A" if zA < zB else "B"
        ik = SIDES[side]["knee"]
        (dst / subj).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / subj / "videos" / cam / "deadlift.mp4"))
        writers, meta = {}, dict(vid_frame=[], gt2d=[], hidden=[], disc_c=[], disc_r=[])
        i = 0
        while i < end:
            if i < start or (i - start) % STRIDE:
                cap.grab()
                i += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            if not writers:
                h, w = img.shape[:2]
                for name in ("clean", "hidden"):
                    writers[name] = cv2.VideoWriter(str(dst / subj / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            uv, z = project(J[i], R, T, K, dist)
            r = float(K[0, 0] * PLATE_RADIUS_M / z[ik])
            hid = b <= i < b + BLOCK
            writers["clean"].write(img)
            writers["hidden"].write(draw_disc(img, uv[ik], r) if hid else img)
            meta["vid_frame"].append(i); meta["gt2d"].append(uv); meta["hidden"].append(hid)
            meta["disc_c"].append(uv[ik]); meta["disc_r"].append(r)
            i += 1
        cap.release()
        for wr in writers.values():
            wr.release()
        np.savez(dst / subj / "meta.npz", side=side, block_start=b, **{k: np.array(v) for k, v in meta.items()})
        n = len(meta["vid_frame"])
        print(f"{subj}: window {start}-{end}, {n} frames at {FPS} fps, hidden {int(np.sum(meta['hidden']))}; "
              f"{nbent} of {nsamp} saved samples in the block are bent over", flush=True)
    with zipfile.ZipFile("outputs/fit3d_track.zip", "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(q for s in subjects for q in (dst / s).rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(dst))
    print(f"zip {Path('outputs/fit3d_track.zip').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["s04", "s05", "s08", "s10"])
    ap.add_argument("--cam", default="60457274")
    a = ap.parse_args()
    main(a.subjects, a.cam)
