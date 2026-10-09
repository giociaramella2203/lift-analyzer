# Lift analyzer: how do pretrained pose models cope with a barbell plate in front of the legs?

> **Status: work in progress (October 2026).** One lifter, four phone-filmed clips, hand labels on 41 frames of two clips. Everything below is an observation on this small dataset, not a validated result.
> Items marked **[TODO]** are planned and not done.

## Summary

**Question.** When a barbell plate hides the knee, what do pretrained pose models do, and does it change the angle measurements built on top of them?

**Setup.** One lifter, four phone-filmed clips (28 reps), hand labels on 17 frames (deadlift with plates) and 24 frames (empty bar). Nothing is trained: two pretrained models (MediaPipe, RTMPose) plus hand-written rep logic.

**What I found (one clip, indicative only).**
- All 28 real reps were detected.
- Behind the plate, RTMPose puts the knee much closer to my labels than MediaPipe (median error 10.0% vs 24.6% of torso length; my own repeat noise is 5.4 to 6.6%).
- That ranking reverses for the hip angle, which the rep analysis uses: MediaPipe 7.6 deg vs RTMPose 14.0 deg behind the plate.
- On the empty-bar clip MediaPipe's knee error is about 2.3 times lower, which points at the plate but is **not established** (the two clips differ in camera distance and resolution).
- A point tracker with temporal memory (CoTracker3) did **not** beat a control that simply freezes the knee at its bottom position.

**Main limitation and next step.** One lifter, small label sets, one labeller, and clips filmed differently. Next: same-camera footage with and without plates, and more labelled frames with the knee hidden.

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
6. `src/pose/track_knee.py`, `src/pose/hold_knee.py`, `src/metrics/show_track.py`: point-tracker experiment (needs a GPU,
   run on Colab), its "hold still" control, and a picture of tracker vs model.
7. `src/metrics/clean_keypoints.py`: bone-length filter. `label_frames.py` / `eval_labels.py`: hand labelling and
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
`eval_labels.py`). Labels are the average of three sessions (A, B, and C a day later, same frames, shuffled).
Errors are medians, as % of the labelled torso length (about 140 to 160 px). My own repeat error between
sessions, over all frames, was 2.4 to 3.4% (shoulder), 3.0 to 4.3% (hip) and 5.4 to 6.6% (knee). That is the noise
floor: knee differences of about that size or smaller cannot be told apart. (I did not separate the hidden frames.)

Both models differ from my clicks by a constant amount (measured with the first two sessions): both put the
shoulder 13 to 21 px higher and the knee 10 to 12 px behind where I click (I click the front of the kneecap, they place the joint centre). This is a
labelling-convention gap, not a plate effect, so the table also shows the error after removing each model's
average offset (estimated on the other frames, leave-one-out):

| Error after removing the average offset | MediaPipe | RTMPose |
|---|---|---|
| Knee position, all frames (% torso) | 11.5 | 6.4 |
| Knee position, knee visible (n=11) | 9.3 | 5.4 |
| Knee position, knee hidden (n=6) | 24.6 | 10.0 |
| Hip angle, all frames (deg) | 4.4 | 8.8 |
| Hip angle, knee visible (deg) | 3.8 | 5.5 |
| Hip angle, knee hidden (deg) | 7.6 | 14.0 |

(Without the offset removal, the hip-angle error is 15.1 deg for MediaPipe and 12.6 deg for RTMPose.)

- RTMPose places the knee closer to my labels than MediaPipe in every subset I looked at, including behind
  the plate (hidden: 10.0 vs 24.6, about 3.7 times my own repeat error for MediaPipe). MediaPipe is slightly
  better on the shoulder.
- That does **not** carry over to the hip angle, which is what the rep analysis uses: MediaPipe's median
  angle error is lower in every subset (4.4 vs 8.8 deg overall, 7.6 vs 14.0 behind the plate), and its 90th
  percentile too (8.5 vs 15.2 deg). With 17 frames, and hand labels that are themselves a guess behind the
  plate, I treat this as a tendency on one clip, not a settled ranking.
- The measured angle error (median 4 to 9 deg, 90th percentile 8 to 15 deg) is comparable to the 15 deg
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

Next to the plate clip, MediaPipe's knee error is about 2.3 times lower overall (11.5 to 4.9) and its
hidden-knee error about 4.6 times lower (24.6 to 5.3). RTMPose changes much less (6.4 to 5.1), and its
hip-angle error is lower in the empty-bar clip (8.8 to 2.5 deg; MediaPipe 4.4 to 1.6). A knee hidden by the bar or shorts is therefore
not what makes MediaPipe fail; the bar is thin, though, so it hides far less of the leg than a plate. That points to the plate, but it is **not established**: the two clips
differ in camera distance and resolution, the sets have 17 and 24 frames, and all labels are mine (set 2 is the average of three sessions, set 1 is a
single session, which if anything makes the plate clip look slightly better than it is). The raw
errors before offset removal are large in both clips (hip angle about 11 to 15 deg), because of the
click-versus-joint-centre convention gap (model hip about 11 to 13 px below my click, knee about 15 to 19 px
to the side).

**[TODO]** More labelled frames on same-camera footage with and without plates.

For comparison, the empty-bar deadlift was about 2.6 times less noisy (0.40 vs 1.05 deg) with far fewer
low-confidence frames (2% vs 44%). That pair is confounded: the two clips differ in camera distance and
resolution, so it does not isolate the effect of the plate.

### 4. Does temporal memory help? (plate clip, 17 labelled frames)

Idea: the plate hides the knee, so give the system a memory: take the knee at a moment when it is visible and carry
it through the frames where it is hidden. I tried a point tracker (CoTracker3, pretrained, run on a Colab GPU) and a
control that just copies the knee position from the bottom of the rep to every frame of that rep (`hold_knee.py`).

- **First attempt failed.** I anchored the tracker on the first frame of each rep. In this camera view the plate
  already covers the knee at lockout, so the tracker was given a point on the plate and followed the plate to
  the floor (median knee error about 185% of torso length). Lesson: the anchor must be a frame where the knee is visible.
- **Second attempt (anchor at the bottom of the rep, tracked both ways)** behaves sensibly: it stays in the knee region
  and reports "not visible" on the frames where the plate crosses the knee.

Error after removing the average offset, medians (laid out as in section 3):

| | Knee, all (% torso) | Knee, hidden (n=6) | Hip angle, all (deg) | Hip angle, hidden (deg) |
|---|---|---|---|---|
| MediaPipe | 11.5 | 24.6 | 4.4 | 7.6 |
| RTMPose | 6.4 | 10.0 | 8.8 | 14.0 |
| Point tracker (anchored at the bottom) | 6.5 | 10.9 | 4.5 | 7.0 |
| Control: knee held still at its bottom position | 5.3 | 5.6 | 4.2 | 7.2 |

**The tracker did not beat the control.** On the knee the tracker is no better than per-frame RTMPose (6.5 vs 6.4)
and worse than the control (5.3), and its 90th-percentile knee error is higher (14.9 vs 9.6 overall, 19.7 vs 11.2 behind
the plate). On the angle, MediaPipe, the control and the tracker are about equal (4.2 to 4.5 deg overall); RTMPose is
worse (8.8 deg), and since the control and tracker use RTMPose's shoulder and hip, that error comes from its knee point.
In a deadlift the knee barely moves in the image, so a frozen position is a strong baseline.
What does look useful is anchoring on a clear frame (the control beat per-frame RTMPose on the knee: 5.3 vs 6.4 overall,
5.6 vs 10.0 behind the plate), but the hidden-frame gap (4.4) is smaller than my own knee repeat error (5.4 to 6.6),
so it is within noise, even though it points the same way as with the first two sessions. The tracker's design was changed after the first failure, on this
same clip, so these numbers are development results. The "bottom" rows are partly circular, since the anchor frame is
a bottom frame. I would expect memory to matter more for a joint that moves while hidden, which I have not tested.

**[TODO]** Test the frozen protocol, with the control, on new same-camera footage.

### 5. Timing numbers depend on the model

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
