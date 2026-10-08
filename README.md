# Lift analyzer: how do pretrained pose models cope with a barbell plate in front of the legs?

> **Status: work in progress (October 2026).** One lifter, four phone-filmed clips, hand labels on 41 frames of two clips. Everything below is an observation on this small dataset, not a validated result.
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
6. `src/metrics/clean_keypoints.py`: bone-length filter. `label_frames.py` / `eval_labels.py`: hand labelling and
   scoring against the labels.

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

**Trajectories (no labels needed).**

| | MediaPipe | RTMPose |
|---|---|---|
| Frame-to-frame angle noise | 1.05 deg | 0.75 deg |
| Reps detected | 7 | 7 |

- The two models place shoulder, hip and knee within about 1 to 1.5% of image height of each other
  (typically), but their hip-angle curves differ by more than 10 degrees in 34% of frames, and those
  disagreements concentrate in the ascents, when the plate rises past the knees.
- Plotting left and right knee over time (`knee_paths.py`) shows no near/far leg swap. MediaPipe's knee
  makes excursions of roughly 60 to 100 px around each ascent, larger than the gap between the legs;
  RTMPose's are much smaller.
- A bone-length filter (`clean_keypoints.py`) lowers the jitter (MediaPipe 1.05 to 0.85 deg) but does not
  change the error against hand labels, because errors that rotate the thigh keep its length.

![Left and right knee paths](docs/knee_paths_deadlift_plates.png)

**Against hand labels (17 frames from this clip, 6 with the knee hidden by the plate).**
I clicked shoulder, hip and knee on frames chosen mostly from the ascents (`label_frames.py`, scored by
`eval_labels.py`). Errors are medians, as % of the labelled torso length (about 140 to 160 px). My own repeat
error between two labelling sessions was 3.4% (shoulder), 3.1% (hip) and 6.6% (knee).

Both models differ from my clicks by a constant amount: both put the shoulder 13 to 21 px higher and the knee
10 to 12 px behind where I click (I click the front of the kneecap, they place the joint centre). This is a
labelling-convention gap, not a plate effect, so the table also shows the error after removing each model's
average offset (estimated on the other frames, leave-one-out):

| Error after removing the average offset | MediaPipe | RTMPose |
|---|---|---|
| Knee position, all frames (% torso) | 12.2 | 7.4 |
| Knee position, knee visible (n=11) | 10.6 | 6.6 |
| Knee position, knee hidden (n=6) | 22.9 | 8.2 |
| Hip angle, all frames (deg) | 5.1 | 8.5 |
| Hip angle, knee visible (deg) | 4.6 | 4.8 |
| Hip angle, knee hidden (deg) | 7.8 | 14.5 |

(Without the offset removal, the hip-angle error is 15.0 deg for MediaPipe and 12.4 deg for RTMPose.)

- RTMPose places the knee closer to my labels than MediaPipe in every subset I looked at, including behind
  the plate. MediaPipe is slightly better on the shoulder.
- That does **not** carry over to the hip angle, which is what the rep analysis uses: MediaPipe's median
  angle error is lower overall (5.1 vs 8.5 deg), and equal on frames where the knee is visible. With 17
  frames, 90th-percentile errors of 8 to 24 deg, and hand labels that are themselves a guess behind the plate,
  I cannot say which model is better for the angle.
- The measured angle error (median 5 to 8 deg, 90th percentile up to about 15) is comparable to the 15 deg
  tolerance of the rep-depth check, so single-rep depth flags near the threshold are not reliable.

**The same labelling on the empty-bar deadlift (24 frames, no plate, 10 of them marked "knee hidden").**
Same procedure and convention, one labelling session. Here the knee is hidden by the bar or by my shorts
(my own judgement per frame), not by a plate. Error after removing the average offset, medians:

| | MediaPipe | RTMPose |
|---|---|---|
| Knee position, all frames (% torso) | 4.9 | 5.1 |
| Knee position, knee hidden (n=10) | 5.3 | 4.8 |
| Hip angle, all frames (deg) | 1.6 | 2.5 |
| Hip angle, 90th percentile (deg) | 5.4 | 8.3 |

Next to the plate clip, MediaPipe's knee error is about 2.5 times lower overall (12.2 to 4.9) and its
hidden-knee error about 4 times lower (22.9 to 5.3). RTMPose changes much less (7.4 to 5.1), and its
hip-angle error is lower in the empty-bar clip (8.5 to 2.5 deg). A knee hidden by the bar or shorts is therefore
not what makes MediaPipe fail; the bar is thin, though, so it hides far less of the leg than a plate. That points to the plate, but it is **not established**: the two clips
differ in camera distance and resolution, the sets have 17 and 24 frames, and all labels are mine. The raw
errors before offset removal are large in both clips (hip angle about 11 to 13 deg), because of the
click-versus-joint-centre convention gap (model hip about 11 to 13 px below my click, knee about 15 to 19 px
to the side).

**[TODO]** More labelled frames on same-camera footage with and without plates, and a repeat
labelling session on another day for the hidden-knee frames.

For comparison, the empty-bar deadlift was about 2.6 times less noisy (0.40 vs 1.05 deg) with far fewer
low-confidence frames (2% vs 44%). That pair is confounded: the two clips differ in camera distance and
resolution, so it does not isolate the effect of the plate.

### 4. Timing numbers depend on the model

MediaPipe's curve suggested the last reps of the plate set were slower (0.97 and 0.87 s ascent). RTMPose
gives about 0.57 s for the same reps. I attribute this to the misplaced knee distorting the angle near
lockout, and I do not report it as a fatigue effect.

## Limitations

- One lifter, four clips, a single gym, a phone camera. No generalisation is claimed.
- Hand labels exist for 17 frames (plates) and 24 frames (empty bar), from a single labeller. Behind the plate they are an estimate, not a measurement. Agreement between two models is not accuracy.
- Absolute angles depend on where each model puts the joint and on where I click, so only within-set comparisons are made.
- At 30 fps one frame is about 6% of a 0.6 s ascent, so tempo differences of that size are noise.
- Squat knee angles are unreliable: the feet are cropped at the bottom of the frame, so the ankle is
  extrapolated. Squat is used here for rep counting only. **[TODO: re-film with feet in frame]**
- Bench is filmed from floor level and the hips are out of frame, so only the elbow angle is used.
- The hold detection uses a fixed 8 degree band around the bottom angle.

## Planned

- [ ] Same camera position, with and without plates, repeated sets.
- [x] First hand-labelled checks (17 frames with plates, 24 frames empty bar, results above).
- [ ] More labelled frames on several clips, second labelling session on another day for the hidden-knee frames.
- [ ] A set with deliberately planted faults (half rep, paused rep) to test whether the flags catch them.
- [ ] Optional: fine-tune RTMPose on public lifting data, tested on my own labelled frames.

## Related work

Stanford CS231n 2024, *Automating powerlifting judging through keypoint detection*: a similar
pretrained-keypoints-plus-rules approach; its rule-based method outperformed a CNN trained on the keypoints.

## Privacy

Raw video and generated overlays are excluded (`.gitignore`) because they show a person and a gym.
