## Scripts and data

### What the code does

Everything runs on CPU, except the point tracker and the 2D-to-3D lifter, which I ran on a Colab GPU.

1. `src/pose/extract_mediapipe.py`: MediaPipe Pose Landmarker (heavy), 33 landmarks per frame.
2. `src/pose/extract_rtmpose.py`: RTMPose through `rtmlib` (ONNX), 17 COCO keypoints, saved in the same format.
3. `src/metrics/lift_reps.py`: joint-angle curve per lift (squat: knee, deadlift: hip, bench: elbow),
   rep segmentation, and a separate "hold" phase so a pause at the bottom is not counted as slow lifting.
4. `src/metrics/hud_video.py`: rep counter and a one-line check per rep drawn on the video.
5. `src/metrics/jitter.py`, `compare_models.py`, `knee_paths.py`: noise estimate, MediaPipe vs RTMPose
   comparison, and left/right knee trajectories.
6. `src/pose/track_knee.py`, `src/pose/hold_knee.py`, `src/metrics/show_track.py`: point-tracker experiment (needs a GPU,
   run on Colab), its "hold still" control, and a picture of tracker vs model.
7. `src/metrics/clean_keypoints.py`: bone-length filter. `label_frames.py` / `eval_labels.py`: hand labelling and
   scoring against the labels.
8. 3D checks: `src/metrics/world3d.py` (MediaPipe's 3D world landmarks against the labels), `src/pose/export_h36m.py`
   (RTMPose keypoints to the 17-joint format a lifter expects), `notebooks/lift3d_motionbert.ipynb` (MotionBERT-Lite on Colab) and
   `src/metrics/eval_lift3d.py` (the lifted result against the labels).
9. Occlusion checks: `src/metrics/eval_masking.py` (oracle masking of the knee), `src/metrics/occlusion_signals.py` (label-free
   signals for "knee hidden", AUC), `src/metrics/eval_limb_repair.py` (limb-length repair vs interpolation),
   `src/metrics/eval_detect_repair.py` (detect, then repair, threshold tuned on one clip and tested on the other),
   `src/metrics/eval_gap.py` (hidden-minus-visible error, the main metric; `--sessions` scores label sessions separately) and
   `src/metrics/plot_gap.py` (the figure in the README). Hand-labelled frames live in `labels/`; `labels/strict/` holds a
   pass done with a stricter written rule and is kept out of the default evaluation.

```
python src/pose/extract_mediapipe.py data/raw/<clip>.mp4
python src/metrics/lift_reps.py outputs/<clip>/mediapipe_keypoints.npz --lift deadlift
python src/metrics/hud_video.py <clip> --lift deadlift
python src/pose/extract_rtmpose.py data/raw/<clip>.mp4 --mode balanced
python src/metrics/compare_models.py <clip> --lift deadlift
python src/metrics/knee_paths.py <clip>
```

Nothing is trained or fine-tuned. The pose models are pretrained; the rep logic is hand-written
signal processing (Savitzky-Golay smoothing, peak finding, fixed thresholds).

### Data

Four clips of one lifter, shot on a phone from the side in a gym: bodyweight squat (5 reps), bench
press (8), Romanian deadlift with an empty bar (8) and with plates (7). Footage is not included in
this repository.
