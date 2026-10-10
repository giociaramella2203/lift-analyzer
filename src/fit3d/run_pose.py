"""Fit3D controlled occlusion experiment, step 1: run the pose models on clean and disc-occluded frames.

Usage (from the repository root, inside the .venv that has rtmlib and mediapipe):
  python src/fit3d/run_pose.py --subjects s05 --limit 10        # quick test (10 frames)
  python src/fit3d/run_pose.py                                   # all 8 train subjects, 2 cameras
Resumable: a finished (subject, camera) file in outputs/fit3d/ is skipped.

What it does, per subject and camera, on the deadlift video:
  * projects the 25 true 3D joints into the image (X_cam = (X - T) @ R.T, pinhole + lens distortion; verified on an overlay);
  * takes every --step-th frame between the first and last repetition boundary (rep_ann.json);
  * runs RTMPose (balanced) and MediaPipe (heavy, one image at a time, no tracking) on three versions of each frame:
      clean    the frame as is
      hidden   an opaque grey disc, the size of a 45 cm plate at the knee's depth, centred on the TRUE projected knee
      control  the same disc moved sideways, away from the body, so the knee stays visible (separates "the knee is hidden"
               from "there is a disc in the picture")
  * saves the 2D predictions, the projected true joints and the disc geometry. No frames are saved except a few preview
    images in outputs/fit3d/ (git-ignored; Fit3D may not be redistributed).
The occluded leg is the one nearer the camera. Joint order (found from an overlay): hip/knee/ankle = 1/2/3 and 4/5/6,
shoulders 11 and 14, pelvis 0.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pose"))

ROOT = Path("data/fit3d/train")
OUT = Path("outputs/fit3d")
SIDES = {"A": dict(shoulder=11, hip=1, knee=2), "B": dict(shoulder=14, hip=4, knee=5)}
PLATE_RADIUS_M = 0.225        # a 45 cm competition plate
CONDS = ("clean", "hidden", "control")


def load_cam(subj, cam):
    d = json.load(open(ROOT / subj / "camera_parameters" / cam / "deadlift.json"))
    R = np.array(d["extrinsics"]["R"], float)
    T = np.array(d["extrinsics"]["T"], float).reshape(-1)
    iw = d["intrinsics_w_distortion"]
    f, c, k, p = (np.array(iw[n], float).reshape(-1) for n in ("f", "c", "k", "p"))
    K = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]])
    dist = np.array([k[0], k[1], p[0], p[1], k[2] if len(k) > 2 else 0.0])
    return R, T, K, dist


def project(X, R, T, K, dist):
    Xc = (X - T) @ R.T
    uv, _ = cv2.projectPoints(Xc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2), Xc[:, 2]


def draw_disc(img, c, r):
    out = img.copy()
    c = (int(round(c[0])), int(round(c[1])))
    r = int(round(r))
    cv2.circle(out, c, r, (48, 48, 48), -1, cv2.LINE_AA)
    cv2.circle(out, c, r, (95, 95, 95), 3, cv2.LINE_AA)
    cv2.circle(out, c, max(r // 6, 2), (75, 75, 75), -1, cv2.LINE_AA)
    return out


def main(subjects, cams, step, limit, mode, mp_model):
    from rtmlib import Body
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from extract_rtmpose import COCO_TO_MP, pick_person

    rtm = Body(mode=mode, backend="onnxruntime", device="cpu")
    lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(mp_model)), running_mode=vision.RunningMode.IMAGE))

    def run_rtm(img):
        kp, sc = rtm(img)
        row = np.full((33, 3), np.nan)
        k = pick_person(kp, sc) if len(kp) else None
        if k is not None:
            for c, m in COCO_TO_MP.items():
                row[m] = (kp[k][c][0], kp[k][c][1], sc[k][c])
        return row

    def run_mp(img):
        h, w = img.shape[:2]
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not res.pose_landmarks:
            return np.full((33, 3), np.nan)
        return np.array([[p.x * w, p.y * h, p.visibility] for p in res.pose_landmarks[0]])

    OUT.mkdir(parents=True, exist_ok=True)
    for subj in subjects:
        J = np.array(json.load(open(ROOT / subj / "joints3d_25" / "deadlift.json"))["joints3d_25"], float)
        reps = json.load(open(ROOT / subj / "rep_ann.json"))["deadlift"]
        lo, hi = int(reps[0]), min(int(reps[-1]), len(J) - 1)
        frames = np.arange(lo, hi + 1, step)
        if limit:
            frames = frames[:limit]
        for cam in cams:
            dst = OUT / f"{subj}_{cam}{'_test' if limit else ''}.npz"
            if dst.exists():
                print("skip", dst.name)
                continue
            R, T, K, dist = load_cam(subj, cam)
            # near side = the hip with the smaller mean depth
            zA = ((J[:, SIDES["A"]["hip"]] - T) @ R.T)[:, 2].mean()
            zB = ((J[:, SIDES["B"]["hip"]] - T) @ R.T)[:, 2].mean()
            side, other = ("A", "B") if zA < zB else ("B", "A")
            ik, io = SIDES[side]["knee"], SIDES[other]["knee"]
            cap = cv2.VideoCapture(str(ROOT / subj / "videos" / cam / "deadlift.mp4"))
            want = set(frames.tolist())
            res = {f"{m}_{c}": [] for m in ("rtm", "mp") for c in CONDS}
            gt, depth, disc_c, disc_r, ctrl_c, got = [], [], [], [], [], []
            t0, i = time.time(), 0
            last = int(frames.max())
            while i <= last:
                if i not in want:
                    cap.grab()
                    i += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                uv, z = project(J[i], R, T, K, dist)
                r = float(K[0, 0] * PLATE_RADIUS_M / z[ik])
                sgn = 1.0 if uv[ik, 0] >= uv[io, 0] else -1.0
                cc = uv[ik] + np.array([sgn * 2.7 * r, 0.0])
                imgs = {"clean": img, "hidden": draw_disc(img, uv[ik], r), "control": draw_disc(img, cc, r)}
                for c in CONDS:
                    res[f"rtm_{c}"].append(run_rtm(imgs[c]))
                    res[f"mp_{c}"].append(run_mp(imgs[c]))
                gt.append(uv); depth.append(z); disc_c.append(uv[ik]); disc_r.append(r); ctrl_c.append(cc); got.append(i)
                if len(got) == 1 or len(got) % 20 == 0:
                    print(f"{subj} {cam} frame {i} ({len(got)}/{len(frames)})  {time.time() - t0:.0f}s", flush=True)
                if len(got) == max(len(frames) // 2, 1):
                    x0, y0 = np.clip((uv.min(0) - 60).astype(int), 0, None)
                    x1, y1 = (uv.max(0) + 60).astype(int)
                    sheet = np.hstack([imgs[c][y0:y1, x0:x1] for c in CONDS])
                    cv2.imwrite(str(OUT / f"{subj}_{cam}_preview.jpg"), sheet)
                i += 1
            cap.release()
            np.savez(dst, frames=np.array(got), gt2d=np.array(gt), depth=np.array(depth), disc_c=np.array(disc_c),
                     disc_r=np.array(disc_r), ctrl_c=np.array(ctrl_c), side=side, **{k: np.array(v) for k, v in res.items()})
            print(f"saved {dst.name}: {len(got)} frames, near side {side}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"])
    ap.add_argument("--cams", nargs="*", default=["60457274", "58860488"])
    ap.add_argument("--step", type=int, default=6, help="use every n-th frame (50 fps video)")
    ap.add_argument("--limit", type=int, default=0, help="only the first n sampled frames (quick test)")
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--mp-model", default="models/pose_landmarker_heavy.task")
    a = ap.parse_args()
    main(a.subjects, a.cams, a.step, a.limit, a.mode, a.mp_model)
