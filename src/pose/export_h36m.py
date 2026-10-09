"""Convert saved RTMPose keypoints into the 17-joint Human3.6M layout used as input by 2D-to-3D lifters.

Usage (from the repository root):
  python src/pose/export_h36m.py <clip>        # reads outputs/<clip>_rtm/mediapipe_keypoints.npz
Writes outputs/<clip>_rtm/h36m_2d.npz with
  kp2d   (T, 17, 2)  pixel coordinates, missing frames filled by linear interpolation
  score  (T, 17)     RTMPose keypoint score (0 where it was interpolated)
  motionbert_input (T, 17, 3)  x, y normalised as MotionBERT expects, plus score
  hidden_frames    frames where the knee was marked hidden in the hand labels (for the masking experiment)
  fps, width, height

Joint order (H36M): 0 pelvis, 1 R hip, 2 R knee, 3 R ankle, 4 L hip, 5 L knee, 6 L ankle, 7 spine, 8 thorax,
9 neck/nose, 10 head, 11 L shoulder, 12 L elbow, 13 L wrist, 14 R shoulder, 15 R elbow, 16 R wrist.
Pelvis = mid-hips, thorax = mid-shoulders, spine = mid pelvis/thorax. The extractor stores no eye keypoints,
so the nose is used for both neck/nose (9) and head (10). That makes the head a rough guess; the lifter is
not asked about the head here. Screen-coordinate normalisation is left to the lifter step.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# MediaPipe-33 slot (where the extractor copied the COCO points) for the joints we can take directly
MP = dict(nose=0, ls=11, rs=12, le=13, re=14, lw=15, rw=16, lh=23, rh=24, lk=25, rk=26, la=27, ra=28)


def fill(x):
    """Linear interpolation of NaN gaps in a (T,) array."""
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    return np.interp(idx, idx[ok], x[ok])


def main(clip):
    d = Path("outputs") / f"{clip}_rtm"
    z = np.load(d / "mediapipe_keypoints.npz")
    k, T = z["keypoints"], len(z["keypoints"])
    xy, sc = k[:, :, :2], k[:, :, 2]
    g = lambda n: xy[:, MP[n]]
    s = lambda n: sc[:, MP[n]]
    pelvis = (g("lh") + g("rh")) / 2
    thorax = (g("ls") + g("rs")) / 2
    spine = (pelvis + thorax) / 2
    j = [pelvis, g("rh"), g("rk"), g("ra"), g("lh"), g("lk"), g("la"), spine, thorax, g("nose"), g("nose"),
         g("ls"), g("le"), g("lw"), g("rs"), g("re"), g("rw")]
    scs = [np.minimum(s("lh"), s("rh")), s("rh"), s("rk"), s("ra"), s("lh"), s("lk"), s("la"),
           np.minimum.reduce([s("lh"), s("rh"), s("ls"), s("rs")]), np.minimum(s("ls"), s("rs")), s("nose"), s("nose"),
           s("ls"), s("le"), s("lw"), s("rs"), s("re"), s("rw")]
    kp2d = np.stack(j, axis=1)            # (T, 17, 2)
    score = np.stack(scs, axis=1)         # (T, 17)
    missing = np.isnan(kp2d).any(axis=2)  # (T, 17)
    for jj in range(17):
        for c in range(2):
            kp2d[:, jj, c] = fill(kp2d[:, jj, c])
    score = np.where(missing | np.isnan(score), 0.0, score)
    # MotionBERT's in-the-wild input: x,y centred on the image and divided by half the SHORTER side
    # (so roughly in [-1, 1]), plus a confidence channel. RTMPose scores are not on AlphaPose's scale: if the
    # lifter looks off, run it without the confidence channel (its --no_conf option) or set this column to 1.
    w, h = float(z["width"]), float(z["height"])
    half = min(w, h) / 2
    norm = np.concatenate([(kp2d - np.array([w / 2, h / 2])) / half, np.clip(score, 0, 1)[..., None]], axis=2)  # (T,17,3)
    # frames where I marked the knee as hidden (from the hand labels), for the masking experiment; empty if no labels
    hidden_frames = np.array([], dtype=int)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
        from eval_labels import load_labels
        lab = load_labels(clip).groupby("frame").knee_hidden.max()
        hidden_frames = lab[lab == 1].index.to_numpy(dtype=int)
    except SystemExit:
        print("no labels found: hidden_frames is empty")
    np.savez(d / "h36m_2d.npz", kp2d=kp2d, score=score, motionbert_input=norm, hidden_frames=hidden_frames,
             fps=z["fps"], width=z["width"], height=z["height"])
    print(f"hidden-knee labelled frames: {hidden_frames.tolist()}")
    print(f"{clip}: {T} frames, frames with any missing joint before interpolation: {int(missing.any(axis=1).sum())}")
    print(f"mean score {score.mean():.2f}; saved {d / 'h36m_2d.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    main(ap.parse_args().clip)
