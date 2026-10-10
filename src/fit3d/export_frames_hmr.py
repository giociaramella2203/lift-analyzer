"""Export clean and disc-occluded frames (and person boxes) for a Colab run of a 3D-body-prior model (HMR2.0 / 4D-Humans).

Usage (from the repository root):
  python src/fit3d/export_frames_hmr.py                  # held-out subjects s07-s11, camera 60457274
Writes outputs/fit3d_hmr/<subject>/{clean,hidden}_<frame>.jpg, outputs/fit3d_hmr/boxes.json and outputs/fit3d_hmr.zip.
Frames and zip stay in outputs/ (git-ignored): Fit3D must not be redistributed, so upload the zip only to your own Colab session.
Frames: every 2nd frame processed by run_pose.py (every 12th video frame), the same disc as the 'hidden' version there.
Person box: the true projected joints with 25% padding, so the detector is taken out of the comparison (the fixed-box test showed
that the detector box is not what the disc disturbs).
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_pose import OUT, ROOT, draw_disc  # noqa: E402


def main(subjects, cam, every):
    dst = Path("outputs/fit3d_hmr")
    boxes = {}
    for subj in subjects:
        z = np.load(OUT / f"{subj}_{cam}.npz")
        idx = np.arange(0, len(z["frames"]), every)
        want = {int(z["frames"][i]): i for i in idx}
        (dst / subj).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / subj / "videos" / cam / "deadlift.mp4"))
        i = 0
        while i <= max(want):
            if i not in want:
                cap.grab()
                i += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            j = want[i]
            gt = z["gt2d"][j]
            lo, hi = gt.min(0), gt.max(0)
            pad = 0.25 * (hi - lo)
            box = np.concatenate([np.clip(lo - pad, 0, None), np.minimum(hi + pad, [img.shape[1], img.shape[0]])])
            boxes[f"{subj}/{i}"] = box.round(1).tolist()
            cv2.imwrite(str(dst / subj / f"clean_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(dst / subj / f"hidden_{i}.jpg"), draw_disc(img, z["disc_c"][j], z["disc_r"][j]), [cv2.IMWRITE_JPEG_QUALITY, 95])
            i += 1
        cap.release()
        print(subj, "done", flush=True)
    json.dump(boxes, open(dst / "boxes.json", "w"))
    with zipfile.ZipFile("outputs/fit3d_hmr.zip", "w", zipfile.ZIP_STORED) as zf:
        for p in sorted(dst.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(dst))
    print(f"{len(boxes)} frames x 2 versions; zip {Path('outputs/fit3d_hmr.zip').stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["s07", "s08", "s09", "s10", "s11"])
    ap.add_argument("--cam", default="60457274")
    ap.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    main(a.subjects, a.cam, a.every)
