"""Track the barbell (via the plate seen from the side) with an OpenCV CSRT tracker.

Usage:
    python src/bar/track_bar.py data/raw/<video>.mp4
    python src/bar/track_bar.py data/raw/<video>.mp4 --plate-diam-m 0.45

A window opens on the first frame: draw a box tightly around the plate
(its outer circle), press ENTER to confirm. The centre of the box is
tracked through the video. Outputs go to outputs/<video_name>/:
bar_path.csv, bar_annotated.mp4, bar_path.png.

If a window cannot be used, pass the box in original pixels instead:
    --bbox x,y,w,h
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pick_box(frame, max_h=900):
    """Let the user draw a box on a downscaled copy; return it in original pixels."""
    scale = min(1.0, max_h / frame.shape[0])
    disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    print("Draw a box tightly around the plate, then press ENTER (or SPACE).")
    x, y, w, h = cv2.selectROI("Select the plate", disp, showCrosshair=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        raise SystemExit("No box selected.")
    return tuple(int(round(v / scale)) for v in (x, y, w, h))


def track(video, bbox, out_dir, plate_diam_m):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    if not ok:
        raise FileNotFoundError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h, w = frame.shape[:2]

    if bbox is None:
        bbox = pick_box(frame)
    print(f"Initial box (x, y, w, h): {bbox}")

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / "bar_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows, trail = [], []
    i = 0
    while True:
        if i == 0:
            found, box = True, bbox
        else:
            ok, frame = cap.read()
            if not ok:
                break
            found, box = tracker.update(frame)

        if found:
            x, y, bw, bh = [float(v) for v in box]
            cx, cy = x + bw / 2, y + bh / 2
            trail.append((int(cx), int(cy)))
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (255, 200, 0), 2)
        else:
            cx = cy = np.nan
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(frame, a, b, (0, 255, 255), 3)

        rows.append(dict(frame=i, time_s=i / fps, x_px=cx, y_px=cy, tracked=bool(found)))
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "bar_path.csv", index=False)

    lost = int((~df.tracked).sum())
    x, y = df.x_px.to_numpy(), df.y_px.to_numpy()
    # scale: assumes the initial box height equals the plate diameter
    m_per_px = plate_diam_m / bbox[3] if plate_diam_m else None
    unit, k = ("m", m_per_px) if m_per_px else ("px", 1.0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot((x - np.nanmean(x)) * k, -(y - np.nanmean(y)) * k)
    ax[0].set_xlabel(f"Horizontal position ({unit})")
    ax[0].set_ylabel(f"Height ({unit})")
    ax[0].set_title("Bar path (side view)")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].plot(df.time_s, -(y - np.nanmean(y)) * k)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel(f"Height ({unit})")
    ax[1].set_title("Bar height over time")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_path.png", dpi=150)

    print(f"{len(df)} frames, bar lost in {lost} frames.")
    print(f"Horizontal range of bar: {np.nanmax(x) - np.nanmin(x):.0f} px"
          + (f" = {(np.nanmax(x) - np.nanmin(x)) * m_per_px * 100:.1f} cm" if m_per_px else ""))
    print(f"Vertical range of bar:   {np.nanmax(y) - np.nanmin(y):.0f} px"
          + (f" = {(np.nanmax(y) - np.nanmin(y)) * m_per_px * 100:.1f} cm" if m_per_px else ""))
    print(f"Saved bar_path.csv, bar_annotated.mp4, bar_path.png to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--bbox", help="x,y,w,h in original pixels (skips the window)")
    ap.add_argument("--plate-diam-m", type=float, default=None,
                    help="plate diameter in metres (0.45 for a standard plate) to get real units")
    a = ap.parse_args()
    bbox = tuple(int(v) for v in a.bbox.split(",")) if a.bbox else None
    track(a.video, bbox, Path("outputs") / Path(a.video).stem, a.plate_diam_m)