"""Is the drift of the shoulder and hip caused by the disc also hiding the arms?  Step 1: re-run RTMPose with other disc placements.

Usage (from the repository root, inside the .venv):
  python src/fit3d/run_cover.py --subjects s05 --cams 60457274 --limit 10      # quick test
  python src/fit3d/run_cover.py                                                  # all files from run_pose.py (RTMPose only, roughly 15-20 minutes)
Resumable: finished files (outputs/fit3d/cover_<subject>_<camera>.npz) are skipped.

Uses every 2nd frame already processed by run_pose.py (same frames, same clean reference).  Two new disc placements:
  knee_half     a disc centred on the true knee with HALF the plate radius (the knee is still hidden, fewer neighbouring joints are)
  wrists_full   a plate-sized disc centred between the two true wrists (the hands and forearms are hidden, the knee is not centred under it)
The original run's 'hidden' version (plate-sized disc on the knee) is the third condition and is reused by eval_cover.py.
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

WRISTS = (13, 16)
CONDS = ("knee_half", "wrists_full")


def main(subjects, cams, limit, mode, every):
    from rtmlib import Body
    from extract_rtmpose import COCO_TO_MP, pick_person

    body = Body(mode=mode, backend="onnxruntime", device="cpu")

    def run_rtm(img):
        kp, sc = body(img)
        row = np.full((33, 3), np.nan)
        k = pick_person(kp, sc) if len(kp) else None
        if k is not None:
            for c, m in COCO_TO_MP.items():
                row[m] = (kp[k][c][0], kp[k][c][1], sc[k][c])
        return row

    for subj in subjects:
        for cam in cams:
            src = OUT / f"{subj}_{cam}.npz"
            dst = OUT / f"cover_{subj}_{cam}{'_test' if limit else ''}.npz"
            if dst.exists():
                print("skip", dst.name)
                continue
            if not src.exists():
                print("missing", src.name, "(run run_pose.py first)")
                continue
            z = np.load(src)
            idx = np.arange(0, len(z["frames"]), every)
            if limit:
                idx = idx[:limit]
            want = {int(z["frames"][i]): k for k, i in enumerate(idx)}
            cap = cv2.VideoCapture(str(ROOT / subj / "videos" / cam / "deadlift.mp4"))
            centre = {c: np.zeros((len(idx), 2)) for c in CONDS}
            radius = {c: np.zeros(len(idx)) for c in CONDS}
            res = {c: [None] * len(idx) for c in CONDS}
            t0, i = time.time(), 0
            while i <= max(want):
                if i not in want:
                    cap.grab()
                    i += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                k = want[i]
                j = idx[k]
                gt = z["gt2d"][j]
                centre["knee_half"][k], radius["knee_half"][k] = z["disc_c"][j], 0.5 * z["disc_r"][j]
                centre["wrists_full"][k], radius["wrists_full"][k] = gt[list(WRISTS)].mean(axis=0), z["disc_r"][j]
                for c in CONDS:
                    res[c][k] = run_rtm(draw_disc(img, centre[c][k], radius[c][k]))
                if k == 0 or (k + 1) % 20 == 0:
                    print(f"{subj} {cam} {k + 1}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
                if k == len(idx) // 2:
                    x0, y0 = np.clip((gt.min(0) - 60).astype(int), 0, None)
                    x1, y1 = (gt.max(0) + 60).astype(int)
                    sheet = np.hstack([draw_disc(img, centre[c][k], radius[c][k])[y0:y1, x0:x1] for c in CONDS])
                    cv2.imwrite(str(OUT / f"{subj}_{cam}_cover_preview.jpg"), sheet)
                i += 1
            cap.release()
            np.savez(dst, idx=idx, **{f"rtm_{c}": np.array(res[c]) for c in CONDS},
                     **{f"centre_{c}": centre[c] for c in CONDS}, **{f"radius_{c}": radius[c] for c in CONDS})
            print(f"saved {dst.name}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"])
    ap.add_argument("--cams", nargs="*", default=["60457274", "58860488"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("--every", type=int, default=2, help="use every n-th processed frame")
    a = ap.parse_args()
    main(a.subjects, a.cams, a.limit, a.mode, a.every)
