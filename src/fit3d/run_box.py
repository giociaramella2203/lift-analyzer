"""Does the disc change the person box that the detector hands to RTMPose?  Step 1: re-run RTMPose with the box held fixed.

Usage (from the repository root, inside the .venv):
  python src/fit3d/run_box.py --subjects s05 --cams 60457274 --limit 10     # quick test
  python src/fit3d/run_box.py                                                 # all files from run_pose.py (RTMPose only, roughly 15 minutes)
Resumable: finished files (outputs/fit3d/box_<subject>_<camera>.npz) are skipped.

RTMPose is two stages: a detector (YOLOX) finds the person box, then the pose model reads the keypoints inside that box (the box
sets the crop and the scale). The disc is drawn on the image BEFORE the detector runs, so it may change the box. For every frame
already processed by run_pose.py (same frames, same disc position) this saves:
  box_clean, box_hidden           the detector's largest person box on the clean and on the disc image (x1, y1, x2, y2)
  fixed_clean, fixed_hidden       keypoints (MediaPipe-33 slot layout, as run_pose.py) when the pose model is given the CLEAN box for both
Then eval_box.py compares the hip-angle change with the box free (original) and with the box held fixed.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pose"))
sys.path.insert(0, str(Path(__file__).parent))
from run_pose import OUT, ROOT, draw_disc  # noqa: E402


def largest(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    boxes = np.asarray(boxes, float)[:, :4]
    return boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]


def main(subjects, cams, limit, mode):
    from rtmlib import Body
    from extract_rtmpose import COCO_TO_MP

    body = Body(mode=mode, backend="onnxruntime", device="cpu")

    def pose_in_box(img, box):
        row = np.full((33, 3), np.nan)
        if box is None:
            return row
        kp, sc = body.pose_model(img, bboxes=[box])
        for c, m in COCO_TO_MP.items():
            row[m] = (kp[0][c][0], kp[0][c][1], sc[0][c])
        return row

    for subj in subjects:
        for cam in cams:
            src = OUT / f"{subj}_{cam}.npz"
            dst = OUT / f"box_{subj}_{cam}{'_test' if limit else ''}.npz"
            if dst.exists():
                print("skip", dst.name)
                continue
            if not src.exists():
                print("missing", src.name, "(run run_pose.py first)")
                continue
            z = np.load(src)
            frames = z["frames"][:limit] if limit else z["frames"]
            want = {int(f): i for i, f in enumerate(frames)}
            cap = cv2.VideoCapture(str(ROOT / subj / "videos" / cam / "deadlift.mp4"))
            res = {k: [None] * len(frames) for k in ("box_clean", "box_hidden", "fixed_clean", "fixed_hidden")}
            t0, i = time.time(), 0
            while i <= int(frames.max()):
                if i not in want:
                    cap.grab()
                    i += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                j = want[i]
                hid = draw_disc(img, z["disc_c"][j], z["disc_r"][j])
                bc, bh = largest(body.det_model(img)), largest(body.det_model(hid))
                res["box_clean"][j] = np.full(4, np.nan) if bc is None else bc
                res["box_hidden"][j] = np.full(4, np.nan) if bh is None else bh
                res["fixed_clean"][j] = pose_in_box(img, bc)
                res["fixed_hidden"][j] = pose_in_box(hid, bc)
                if j == 0 or (j + 1) % 20 == 0:
                    print(f"{subj} {cam} {j + 1}/{len(frames)}  {time.time() - t0:.0f}s", flush=True)
                i += 1
            cap.release()
            np.savez(dst, frames=frames, **{k: np.array(v) for k, v in res.items()})
            print(f"saved {dst.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"])
    ap.add_argument("--cams", nargs="*", default=["60457274", "58860488"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="balanced")
    a = ap.parse_args()
    main(a.subjects, a.cams, a.limit, a.mode)
