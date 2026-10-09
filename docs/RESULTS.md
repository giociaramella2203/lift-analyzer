# Detailed results

Appendix to the [README](../README.md); the section numbers are referenced from there.
Everything comes from one lifter, four phone clips filmed from the side, and hand labels on 17 frames of the plate clip
(three sessions, plus a fourth pass with a written rule) and 24 frames of the empty-bar clip (one session). Read all numbers as indicative.

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

![Left and right knee paths](knee_paths_deadlift_plates.png)

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

### 6. Exploratory 3D check (MediaPipe world landmarks)

MediaPipe also outputs 3D "world" landmarks (metres, estimated from the single image, so depth is a model guess).
`world3d.py` asks two questions on the labelled frames of the plate clip (17) and the empty-bar clip (24).
RTMPose has no 3D output here, so this covers MediaPipe only.

- **Bone-length stability.** In 3D the thigh, shank and torso should keep a constant length. On the plate clip their
  median deviation from the clip median is about 3 to 5%, and it is not larger on the frames where I marked the knee as hidden.
  On the empty-bar clip the thigh varies more (about 8%), which I cannot explain.
- **Hip angle against my labels** (median error in degrees, mean bias removed per variant, leave-one-out):

| | Image pixels (2D) | World x,y only | World x,y,z (3D) |
|---|---|---|---|
| Plates, all frames (n=17) | 5.2 | 5.2 | 6.4 |
| Plates, knee hidden (n=6) | 7.6 | 4.9 | 8.8 |
| Empty bar, all frames (n=24) | 2.4 | 3.7 | 6.6 |
| Empty bar, knee hidden (n=10) | 2.0 | 4.4 | 7.3 |

The full 3D angle is not better than the 2D one in any row. The one favourable cell (world x,y, plates, hidden) has 6 frames, so I do not read it as a result.

Is the gap larger than chance? Median of (3D error minus pixel-2D error) over the labelled frames, with a 95% bootstrap interval (resampling frames):

| | World x,y,z vs 2D | World x,y vs 2D |
|---|---|---|
| Plates (n=17) | +2.2 deg [-1.0, +4.2] | +1.2 deg [-1.9, +2.2] |
| Empty bar (n=24) | +5.2 deg [+1.8, +7.1] | +2.1 deg [-0.1, +3.4] |

On the plate clip the 3D angle is **not distinguishable** from the 2D one with 17 frames. On the empty-bar clip it is worse, and the interval excludes zero.

I expected depth to add frame-to-frame noise, but over all frames the 3D angle is not noisier than the 2D one (median change per frame about 1.3 to 1.5 deg in all three variants, `world3d.py` section 4).
If depth is off, it is off consistently rather than jittery. Two untested explanations remain: the 3D angle and my 2D label angle measure different things when
the thigh is not parallel to the image plane (the mean 3D-minus-2D difference is +4.8 and +7.3 deg), or the model's depth is biased for this pose.

**What this does and does not show.** My labels are 2D image points, so a 3D angle is compared with a 2D label angle. There is no 3D ground truth,
which means I can say the 3D angle does not agree better with my 2D labels, but not how accurate the 3D estimate is. From a side camera,
depth is exactly what the model cannot see. These numbers are not comparable with those in section 3, which remove the
average offset on joint positions; here the bias is removed on the angle.

**[TODO]** A way to check 3D that does not rely on my 2D labels (a second camera view, or a dataset with 3D ground truth).

### 7. A real 2D-to-3D lifter on the RTMPose keypoints (MotionBERT-Lite)

To separate "3D does not help" from "MediaPipe's single-frame depth is poor", I fed the RTMPose keypoints (converted to the 17-joint Human3.6M layout by
`export_h36m.py`) to a pretrained temporal lifter, MotionBERT-Lite (`notebooks/lift3d_motionbert.ipynb`, run on Colab; evaluated with `eval_lift3d.py`).
Same labels, same procedure as section 6. Hip-angle error against my labels, degrees, median, mean bias removed (leave-one-out):

| | Plates, all (n=17) | Plates, knee hidden (n=6) | Empty bar, all (n=24) | Empty bar, knee hidden (n=10) |
|---|---|---|---|---|
| RTMPose pixels (2D, the lifter's input) | 8.6 | 12.0 | 2.2 | 1.0 |
| Lifted x,y,z (3D) | 7.6 | 11.7 | 1.5 | 1.7 |
| Lifted minus 2D, 95% bootstrap interval | +0.09 [-0.11, +1.30] | | -0.58 [-1.89, +0.24] | |

- **On the plate clip lifting does not repair the occlusion error.** The 3D angle is not distinguishable from the 2D one (7.6 vs 8.6 deg overall, 11.7 vs 12.0 behind the plate), and it stays well above MediaPipe's 2D error on the same clip (5.2 deg overall, section 6). The knee error that RTMPose makes behind the plate goes into the lifter and comes out again.
- **On the empty-bar clip everything is accurate** (1.5 to 2.2 deg) and the lifted angle is slightly closer to my labels, but the interval includes zero.
- **The lifted 3D angle stays close to its own 2D input** (mean difference +0.7 and +0.1 deg, median absolute difference 2.9 and 3.7 deg), whereas MediaPipe's world landmarks differed from its pixels by a mean of +4.8 and +7.3 deg (section 6). This weakens the idea that the 3D-minus-2D gap in section 6 is just a side-view projection effect, and points more to MediaPipe's depth estimate, which is a single-frame regression. That is a suggestion, not a test: the two pipelines also differ in input keypoints and training.
- The lifted angle is smoother from frame to frame (median change 1.12 vs 1.26 deg on the plate clip, 1.31 vs 1.47 on the empty-bar clip), as expected from a model with a time window, but not closer to my labels where the input was wrong.
- Setting the confidence channel to 1 (instead of RTMPose's score) made the lifted shank length vary more on the plate clip's hidden frames (21% vs 5.5% median deviation, 6 frames). Tentative: it suggests the lifter uses low scores when the knee is covered.

Limits as in section 6: my labels are 2D, so this says nothing about how accurate the 3D itself is; I assume the lifter's x,y are the image-plane axes; 17 and 24 frames; one labeller.

### 8. Oracle masking: can the lifter use time to repair a hidden knee?

RTMPose gives the hidden knee almost the same confidence as a visible one (median score 0.58 vs 0.64 on the plate clip, ranges overlapping), so the lifter has no signal that the knee is lost.
To see whether MotionBERT *could* use temporal context, I marked the left knee as missing in a window of +-K frames (K=5 or 15, about 0.17 or 0.5 s) around each frame where I marked it hidden, and lifted again.
**This uses my labels to say where the knee is lost, so it is an upper bound, not a usable method.** Variants: `conf0` (knee confidence set to 0), `zero` (knee x,y and confidence set to 0),
`interp` (knee x,y replaced by linear interpolation from outside the window), and a lifter-free baseline: the 2D hip angle from RTMPose with the interpolated knee (`eval_masking.py`).
Hip-angle error against my labels, degrees, median, mean bias removed (leave-one-out):

| | Plates, all (n=17) | Plates, hidden (n=6) | Empty bar, all (n=24) | Empty bar, hidden (n=10) |
|---|---|---|---|---|
| RTMPose 2D, no masking | 8.6 | 12.0 | 2.2 | 1.0 |
| Lifted 3D, no masking | 7.6 | 11.7 | 1.5 | 1.7 |
| Lifted, `conf0`, K=15 | 7.4 | 10.2 | 2.2 | 2.6 |
| Lifted, `interp`, K=15 | 6.4 | 10.9 | 2.2 | 2.4 |
| Lifted, `zero`, K=15 | 10.7 | 22.4 | 21.4 | 12.0 |
| **2D, knee interpolated, K=15 (no lifter)** | 6.5 | **7.5** | 3.0 | 2.1 |

- **The lifter barely uses the missing-knee information.** On the plate clip `conf0` and `interp` lower the hidden-frame error by about 1 to 2 degrees (11.7 to 10.2 and 10.9), within noise for 6 frames. Plain 2D interpolation, without any lifter, does better (7.5 degrees), roughly the level of MediaPipe's own 2D on those frames (7.6, section 6). It is the same finding as the "hold the knee still" control in section 4: in a deadlift the knee barely moves, so a simple prior is hard to beat.
- **`zero` is harmful** (22 degrees on the plate clip, 12 to 21 on the empty-bar clip): the model reads (0,0) as a real joint at the image centre, not as "missing". I did not find out how MotionBERT is meant to be given a missing joint, so `conf0` and `zero` are two guesses; a negative result here does not show the lifter cannot use context.
- **On the empty-bar clip masking does nothing useful** (differences within about 0.2 degrees for `conf0` and `interp` at K=5, slightly worse at K=15), as expected: there the knee was not badly misplaced, so removing it only loses information.
- **Why about 7 degrees remain on the hidden frames, whatever is done to the knee.** MediaPipe 2D, the tracker, the hold-still control and interpolation all land between 7.0 and 7.6 degrees there. Taking the knee straight from my labels (model shoulder and hip) still leaves 7.3 degrees on the plate clip's hidden frames (12.0 with the model's knee), and replacing the shoulder or the hip by my labels leaves 6.2 and 6.6. So the floor is not the knee: the hip angle also depends on the model's shoulder and hip, and on the offset between where the model puts them and where I click. My own repeat noise on the angle is small (median difference between sessions about 0.7 degrees on visible and 1.6 on hidden frames), but those sessions share any systematic bias in my guess behind the plate. On the empty-bar clip the same decomposition stays at 1 to 3 degrees, so the plate clip is simply harder for all three points.
- Caveats: oracle placement of the mask, 6 and 10 hidden frames, one lifter, my 2D labels as reference.
- Note added later: the errors above have one mean bias removed over all labelled frames, which blends the model-vs-click gap with the hidden-frame error. Section 9 re-scores the plate effect as hidden minus visible and finds interpolation closes only part of it.

### 9. Can the hidden knee be flagged, and repaired, without my labels?

Sections 4 and 8 used my labels to say where the knee is hidden. Here I ask whether a label-free signal can find those frames, and whether a repair applied only to the flagged frames helps.
Scripts: `occlusion_signals.py`, `eval_limb_repair.py`, `eval_detect_repair.py`. All numbers are on the frames I labelled (plates: 17, of which 6 hidden; empty bar: 24, of which 10 hidden), so every interval is wide.

**Detection signals** (AUC for separating my "hidden" frames from "visible" ones; 0.5 is chance; 95% bootstrap interval over labelled frames; "ascent" = ascent frames only, since the plate hides the knee mostly then and a signal that only tracks movement would otherwise look good):

| Signal (no labels needed) | Plates, all | Plates, ascent only | Empty bar, all | Empty bar, ascent only |
|---|---|---|---|---|
| MediaPipe vs RTMPose knee distance (`disagree`) | 0.74 [0.37, 1.00] | 0.83 | 0.52 [0.28, 0.78] | 0.67 |
| RTMPose thigh/shank length vs clip median (`bone_rtm`) | 0.53 [0.24, 0.81] | 0.12 | 0.76 [0.51, 0.96] | 0.95 |
| MediaPipe thigh/shank length (`bone_mp`) | 0.42 [0.14, 0.73] | 0.17 | n/a (ankle missing in this clip) | n/a |
| RTMPose knee off its 15-frame running median (`path_dev`) | 0.86 [0.55, 1.00] | 0.92 | 0.34 [0.09, 0.61] | 0.49 |
| RTMPose knee speed (`speed`) | 0.77 [0.42, 1.00] | 0.75 | 0.36 [0.14, 0.61] | 0.55 |
| Minus RTMPose knee score (`low_conf`) | 0.85 [0.50, 1.00] | 0.83 | 0.48 [0.24, 0.74] | 0.40 |

- **No signal works on both clips.** `path_dev`, `speed` and `low_conf` rank hidden frames higher on the plate clip and sit at or below chance on the empty-bar clip; `bone_rtm` does the opposite. With 6 and 10 hidden frames, each interval includes 0.5 or comes close to it.
- **The knee score is not completely blind.** Hidden frames have a slightly lower score on the plate clip (median 0.58 vs 0.64, section 8), consistently enough to rank at AUC 0.85, but the ranges overlap, so there is no usable threshold, and on the empty-bar clip it carries nothing.

**Geometric repair.** In the same oracle windows as section 8 (K frames around each hidden frame), the knee is placed where a circle around the hip (median thigh length) meets a circle around the ankle (median shank length), on the side the knee normally bends to, and compared with linear interpolation and with doing nothing. Knee error in units of torso length, hip-angle error in degrees (mean bias removed), medians on the hidden frames:

| | Plates, knee | Plates, angle | Empty bar, knee | Empty bar, angle |
|---|---|---|---|---|
| RTMPose, no repair | 0.18 | 12.0 | 0.11 | 1.0 |
| Interpolation, K=15 | 0.13 | 7.5 | 0.14 | 2.1 |
| Limb-length, K=5 | 0.17 | 8.4 | 0.29 | 9.3 |
| Limb-length, K=15 | 0.17 | 10.1 | 0.29 | 9.3 |

- **On the plate clip the limb-length repair is no better than interpolation** (angle difference limb minus interpolation, K=15: +1.0 degrees, interval [-7.1, +13.4], 6 frames). Both lower the hidden-frame error somewhat, but the intervals include zero.
- **On the empty-bar clip it hurts** (knee error 0.11 to 0.29, interval of the difference [+0.09, +0.29]): the model's knee was already close, so a repair has nothing to fix and the hip and ankle positions it relies on add error.

**Detect, then repair, with no labels at test time.** Flag frames where `disagree` (or `path_dev`) exceeds a threshold tuned on one clip, widen by +-5 frames, interpolate the knee across them, and test on the other clip.
- Tuned on the plate clip, tested on the empty bar: the threshold flags none of the 10 hidden frames (2 of 14 visible); nothing changes.
- Tuned on the empty bar, tested on the plate clip: the threshold flags 97% of all frames (after widening), so the knee path is interpolated from the few frames left. The hip-angle error falls (8.6 to 3.2 over all labelled frames) **while the knee error rises** (0.09 to 0.26). The angle gain is therefore not a better knee; it is a side effect of flattening the path, together with the bias removal in my metric, and I do not count it as an improvement.
- `path_dev` fails in the same way: tuned on the empty bar it flags every frame of the plate clip, and tuned on the plate clip it catches 1 of the 10 hidden empty-bar frames (while flagging 6 of 14 visible ones).

**A cleaner way to score it (hidden minus visible).** The bias-removed error used so far subtracts one mean over all labelled frames, which mixes two things: a roughly constant gap between where the model puts the joints and where I click, and the extra error on hidden frames. `eval_gap.py` reads the constant from the *visible* frames and reports the hidden-frame excess (signed hip-angle error, model minus my label, median hidden minus median visible, degrees, 95% bootstrap interval):

| | Plates: visible | Plates: hidden | Plates: gap | Empty bar: gap |
|---|---|---|---|---|
| RTMPose, no repair | +9.1 | +25.5 | **+16.4** [+8.5, +22.1] | +1.1 [-2.5, +3.6] |
| Interpolation, K=15 | +9.2 | +20.2 | +11.1 [+6.8, +17.8] | +2.2 [-2.1, +5.4] |
| Limb-length, K=5 | +9.1 | +17.7 | +8.5 [-5.0, +15.8] | +14.6 [+6.3, +19.8] |

- **The constant part is large and the same everywhere**: the model's hip angle sits about +9 degrees (plates) and +13 degrees (empty bar) above my clicks on visible frames. That is a definition gap, not occlusion, and it is why absolute angles are not compared.
- **The occlusion effect is the gap**: about 16 degrees on the plate clip (still about 12 degrees, interval [+5.0, +16.6], when only ascent frames are compared, 6 hidden vs 4 visible) and none detectable on the empty bar. This is a cleaner statement of the plate effect than the 12.0 vs 2.2 degrees in section 8, which depended on how the bias was removed.
- **No repair closes the gap.** With the oracle window, interpolation reduces it from 16 to about 11 degrees (interval excludes zero), limb-length to about 9 to 11 (interval includes zero). So the "12.0 to 7.5 degrees" of section 8 flatters interpolation: relative to the visible frames about 11 degrees of excess error remain. On the empty bar the limb-length repair adds a 15 degree gap where there was none.

**Is the constant a labelling convention? (second pass with a written rule).** On the plate clip I re-clicked all 17 frames in a fourth session (D) with an explicit rule (shoulder at the centre of the joint below the bony tip, hip at the greater trochanter, knee at the joint centre) and compared it with sessions A to C. Hidden/visible flags are taken from sessions B and C (session A skipped the hidden frames, and in D I marked almost every frame hidden by mistake, so its own flags are not used; `labels/strict/` keeps that file out of the default evaluation). On the 11 visible frames labelled in all four sessions:
- The constant did not move: +8.7, +8.9, +9.2 and +8.3 degrees in sessions A, B, C and D, although the new rule shifted my shoulder clicks by about 7 px and my hip clicks by about 5 px, more than the 1 to 5 px shifts between the earlier sessions. So, at least for the rules I tried, the +9 degree offset behaves like the model placing the joints differently from where I click, not like a wobble in my clicking.
- The hidden-frame gap stays positive in every session but moves with how I guess the hidden knee: +14.5 [+7.6, +21.0] (B), +19.0 [+8.5, +22.6] (C), +9.4 [+1.8, +19.5] (D), 6 hidden frames. Behind the plate my own reference has an uncertainty of several degrees, so the size of the plate effect is "roughly 10 to 20 degrees", not 16.

**The gap depends on the model.** The same metric for MediaPipe's 2D hip angle, next to RTMPose (signed error in degrees; medians; gap = hidden minus visible with a 95% bootstrap interval, as drawn in `docs/fig_gap.png` by `plot_gap.py`):

| | Plates: visible | Plates: hidden | Plates: gap | Empty bar: gap |
|---|---|---|---|---|
| RTMPose | +9.1 | +25.5 | **+16.4** [+8.5, +22.1] | +1.1 [-2.5, +3.6] |
| MediaPipe | +15.3 | +7.9 | -7.4 [-13.3, +3.2] | +2.8 [-0.0, +5.3] |

On the plate clip MediaPipe's hip angle is not worse on the hidden frames (negative gap, interval includes zero), although its knee position is much worse there (24.6 vs 10.0% of torso length, section 3), and its constant on visible frames is different (+15 vs +9 degrees). So the excess hip-angle error behind the plate is an RTMPose result on this clip, not a property of "pose models" in general, and a good angle does not imply a well-placed knee. With 6 hidden frames I cannot say why MediaPipe's angle holds up.

**What this shows.** On these two clips, none of the cheap label-free signals or repairs I tried gives a reliable gain over leaving RTMPose's knee alone, and the threshold of a detector tuned on one clip does not carry over to the other. Behind the plate the best simple fix is still interpolation with an oracle mask (section 8). Caveats: 6 and 10 hidden frames, one lifter, two clips filmed differently, my 2D labels as reference.

### 10. Retractions

Two results I reported to myself during the work and then withdrew after checking the per-frame errors:
- **Averaging the two models** (MediaPipe and RTMPose) seemed to lower the hip-angle error to about 1.75 degrees. The per-frame signed errors showed that both models share a constant raw offset of about +12 to +14 degrees on the plate clip, and the bias-removed metric hides a constant offset, so it flatters any estimator whose errors cluster tightly. It was a metric artefact, not a gain, and it is not in the results above.
- **A "floor of about 7 degrees"** on hidden frames, whatever was done to the knee (section 8). Much of it was the constant model-versus-click offset mixed with the hidden-frame error by the same bias removal. Section 9 re-scores the effect as hidden minus visible.

## Limitations (full list)

- One lifter, four clips, a single gym, a phone camera. No generalisation is claimed.
- Hand labels exist for 17 frames (plates) and 24 frames (empty bar), from a single labeller. Behind the plate they are an estimate, not a measurement. Agreement between two models is not accuracy.
- Absolute angles depend on where each model puts the joint and on where I click, so only within-set comparisons are made.
- At 30 fps one frame is about 6% of a 0.6 s ascent, so tempo differences of that size are noise.
- Squat knee angles are unreliable: the feet are cropped at the bottom of the frame, so the ankle is
  extrapolated. Squat is used here for rep counting only. **[TODO: re-film with feet in frame]**
- Bench is filmed from floor level and the hips are out of frame, so only the elbow angle is used.
- The hold detection uses a fixed 8 degree band around the bottom angle.
- There is no 3D ground truth: the 3D checks compare a 3D angle with my 2D labels, so they cannot say how accurate the 3D estimate is, and the lifter's x,y axes are assumed to be the image plane.
