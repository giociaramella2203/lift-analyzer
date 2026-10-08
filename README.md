# Lift analyzer: how do pretrained pose models cope with a barbell plate in front of the legs?

> **Status: work in progress (October 2026).** One lifter, four phone-filmed clips, no hand-labelled
> ground truth yet. Everything below is an observation on this small dataset, not a validated result.
> Items marked **[TODO]** are planned and not done.

## Question

Off-the-shelf pose estimators are trained mostly on people in unobstructed poses. In a barbell lift, the
plate and bar pass in front of the body and hide joints that angle-based analysis depends on.
This project asks: **when a plate hides the knee, what does the pose model do, and does it matter for
the measurements built on top of it?**

It started as a small tool for analysing my own squat, bench press and Romanian deadlift videos
(rep counting, joint angles, a per-rep overlay). The failure analysis turned out to be the more
interesting part.

## What the code does

Everything runs on CPU (no GPU needed).

1. `src/pose/extract_mediapipe.py`: MediaPipe Pose Landmarker (heavy), 33 landmarks per frame.
2. `src/pose/extract_rtmpose.py`: RTMPose through `rtmlib` (ONNX), 17 COCO keypoints, saved in the same format.
3. `src/metrics/lift_reps.py`: joint-angle curve per lift (squat: knee, deadlift: hip, bench: elbow),
   rep segmentation, and a separate "hold" phase so a pause at the bottom is not counted as slow lifting.
4. `src/metrics/hud_video.py`: rep counter and a one-line check per rep drawn on the video.
5. `src/metrics/jitter.py`, `compare_models.py`, `knee_paths.py`: noise estimate, MediaPipe vs RTMPose
   comparison, and left/right knee trajectories.

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

## Data

Four clips of one lifter, shot on a phone from the side in a gym: bodyweight squat (5 reps), bench
press (8), Romanian deadlift with an empty bar (8) and with plates (7). Footage is not included in
this repository.

## Results so far

### 1. Rep detection

| Clip | Detected | Real reps | Note |
|---|---|---|---|
| Squat (bodyweight) | 5 | 5 | |
| Bench | 8 | 8 | |
| Deadlift, empty bar | 9 | 8 | the extra detection is the walk-in and bar pick-up at the start |
| Deadlift, plates | 7 | 7 | |

All 28 real reps were found; one extra detection came from the clip starting before I was set up.

### 2. Per-rep consistency check

A rep is flagged if its bottom angle is more than 15 degrees from the set median, or its ascent time is
more than 35% from the median. Thresholds were set before looking at results and not tuned.
"Clean" here means **consistent with the rest of the set, not good technique or safe**.

Of 28 reps, one was flagged: rep 1 of the plate deadlift (about 14 degrees deeper, ascent about 1.75 times
the median) after a roughly 2.5 s grip pause. Both pose models agree it is a slow first pull, and I
confirmed the pause in the video. No other set showed a slowdown, so I make no claim about fatigue.

### 3. Pose models around the plate (deadlift with plates, one clip)

| | MediaPipe | RTMPose |
|---|---|---|
| Frame-to-frame angle noise | 1.05 deg | 0.75 deg |
| Reps detected | 7 | 7 |

- The two models place shoulder, hip and knee within about 1 to 1.5% of image height of each other
  (typically), but their hip-angle curves differ by more than 10 degrees in 34% of frames.
- Those disagreements concentrate in the ascents, when the plate rises past the knees.
- In the frames I inspected, MediaPipe often places the knee off the visible shin line while the plate
  covers it; RTMPose's leg line follows the visible shin better. This is a visual judgement on a handful
  of frames, not a measurement. **[TODO: pixel error against hand-labelled frames]**
- Plotting left and right knee over time (`knee_paths.py`) shows no near/far leg swap. Instead, MediaPipe's
  knee makes excursions of roughly 60 to 100 px around each ascent, larger than the gap between the
  legs, which looks like the hidden joint being guessed. RTMPose shows much smaller excursions.

![Left and right knee paths](docs/knee_paths_deadlift_plates.png)

For comparison, the empty-bar deadlift was about 2.6 times less noisy (0.40 vs 1.05 deg) with far fewer
low-confidence frames (2% vs 44%). That pair is confounded: the two clips differ in camera distance and
resolution, so it does not isolate the effect of the plate. **[TODO: same camera, with and without plates]**

### 4. Timing numbers depend on the model

MediaPipe's curve suggested the last reps of the plate set were slower (0.97 and 0.87 s ascent). RTMPose
gives about 0.57 s for the same reps. I attribute this to the misplaced knee distorting the angle near
lockout, and I do not report it as a fatigue effect.

## Limitations

- One lifter, four clips, a single gym, a phone camera. No generalisation is claimed.
- No ground-truth labels yet. Agreement between two models is not accuracy.
- Absolute angles differ by several degrees between models, so only within-set comparisons are made.
- At 30 fps one frame is about 6% of a 0.6 s ascent, so tempo differences of that size are noise.
- Squat knee angles are unreliable: the feet are cropped at the bottom of the frame, so the ankle is
  extrapolated. Squat is used here for rep counting only. **[TODO: re-film with feet in frame]**
- Bench is filmed from floor level and the hips are out of frame, so only the elbow angle is used.
- The hold detection uses a fixed 8 degree band around the bottom angle.

## Planned

- [ ] Same camera position, with and without plates, repeated sets.
- [ ] Hand-label about 20 frames per clip (knee, hip, shoulder) and report pixel error per model,
      focused on frames where the plate crosses the knee.
- [ ] A set with deliberately planted faults (half rep, paused rep) to test whether the flags catch them.
- [ ] Optional: fine-tune RTMPose on public lifting data, tested on my own labelled frames.

## Related work

Stanford CS231n 2024, *Automating powerlifting judging through keypoint detection*: a similar
pretrained-keypoints-plus-rules approach; its rule-based method outperformed a CNN trained on the keypoints.

## Privacy

Raw video and generated overlays are excluded (`.gitignore`) because they show a person and a gym.
