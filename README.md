# When a joint is hidden: how pretrained pose models fail behind a barbell plate

A small computer-vision study on my own phone videos of deadlifts. Pretrained pose models are the tool I use to measure joint
angles; this repo asks **what they do when a barbell plate hides the knee, whether the measurement built on top of them breaks,
and whether simple fixes help.** It is a failure analysis, not a new method, and the negative results are part of the point.

## In brief

- **Behind the plate, RTMPose's hip angle is off by roughly 10 to 20 degrees more than on frames where the knee is visible**
  (about +16 degrees, 95% interval [+8.5, +22.1], 6 hidden frames). On the empty-bar clip (frames where the bar or my shorts hid the knee) no such excess is detectable (+1 degree).
- **The model does not know it is wrong.** Its knee confidence is almost the same hidden or visible, so the failure cannot be
  caught by looking at the score.
- **The effect depends on the model.** MediaPipe shows no excess in the hip angle on the same frames, although its knee position is
  worse there. A good angle does not mean a well-placed knee.
- **Every simple fix I tried failed or helped only partly:** temporal memory, 3D lifting, interpolation, a limb-length constraint,
  and label-free occlusion detectors. Details below.
- **Small data, one lifter.** 6 to 10 hidden frames per clip, two clips filmed differently, my own hand labels as reference.
  Treat this as a well-posed question with a first answer, not as a result to build on.

![Hip-angle error on visible and knee-hidden frames, both models, both clips](docs/fig_gap.png)

*Each dot is one frame I labelled. Error is the model's hip angle minus my hand-labelled angle. The offset between the
model and my clicks (about +9 to +15 degrees) is roughly constant from frame to frame, so the number that matters is the gap between
the hidden and visible groups.*

## The question

Pose estimators are trained mostly on people in clear view. In a barbell lift the plate and bar pass in front of the legs and hide
the joints that angle-based analysis needs. When the knee is hidden, a pose model still outputs a position. **How wrong is it, can
it be detected, and can it be repaired?** That is the part of the work this repo documents; occlusion robustness and temporal
consistency matter well beyond lifting.

The project began as a tool for my own squat, bench and deadlift videos (rep counting, joint angles, a per-rep consistency
check; all 28 real reps were found). The occlusion question turned out to be the more interesting part.

## What I measured

Setup: one lifter, four clips filmed from the side with a phone (a deadlift set with plates, one with an empty bar, squat and
bench for rep counting). Two pretrained models, MediaPipe Pose and RTMPose. Hand labels of shoulder, hip and knee on 17 frames of
the plate clip (three sessions, plus a fourth pass with a written rule) and 24 frames of the empty-bar clip. Nothing is trained.

1. **The plate effect on the angle.** On the plate clip RTMPose's hip angle is about 16 degrees worse on hidden-knee frames than on
   visible ones (about 12 degrees when only ascent frames are compared). The size depends on how I guess the hidden knee
   (+9 to +19 degrees across my labelling sessions), so I quote it as "roughly 10 to 20".
2. **A roughly constant offset that is not label noise.** RTMPose's angle differs from my clicks by about +9 degrees (plates) to +13
   (bar) on visible frames. It did not move when I re-clicked with a written rule, so it behaves like the model placing joints
   differently from where I click. Scoring hidden minus visible cancels it.
3. **Confidence is uninformative.** Median knee score 0.58 hidden vs 0.64 visible, ranges overlapping, and no signal on the bar clip.

## What I tried to fix it, and what happened

| Approach | Result |
|---|---|
| Swap the model (RTMPose vs MediaPipe) | The excess shows for RTMPose only; MediaPipe's knee is worse but its angle is not |
| Temporal memory (CoTracker3 point tracker, "freeze the knee" control) | The tracker was no better than simply freezing the knee at its bottom position |
| 3D (MediaPipe world landmarks, MotionBERT lifter on RTMPose keypoints) | No better than 2D; the lifter inherits the 2D error. My labels are 2D, so this says nothing about 3D accuracy |
| Oracle interpolation (knee marked missing where I labelled it hidden, then interpolated) | Closes part of the gap (about 16 to 11 degrees); an upper bound, since it uses my labels |
| Limb-length constraint (knee from thigh and shank length) | No better than interpolation, and clearly worse on the bar clip where nothing was hidden |
| Label-free detectors (model disagreement, bone-length deviation, path jumps, confidence) and detect-then-repair | Some separate hidden frames on the plate clip (AUC 0.74 to 0.86, wide intervals) but none works on both clips, and a threshold tuned on one does not carry over |

Two earlier results I retracted after checking: an apparent gain from averaging the two models, and a "7 degree floor" that was
mostly an artefact of how I removed the constant offset. Both are explained in [docs/RESULTS.md](docs/RESULTS.md).

## What this suggests (interpretation, not tested here)

- Pose models are trained to produce a plausible position for hidden joints, so they answer confidently; nothing in their training
  rewards "I cannot see this". That fits the flat confidence scores.
- Fixes that only post-process the output cannot recover information that is not in the 2D input, and repairs such as interpolation can
  damage frames that were fine. A repair is only safe behind a detector that works, and I do not have one.
- The literature points to training-time answers: synthetic occluders (for example BlanketGen2-Fit3D, DAG) and video methods that track
  identity through occlusion (SAM-Body4D, 4DHumans). I read these at abstract level and did not reproduce them.

## How far to trust this

- One lifter, one gym, two clips filmed at different distances and resolutions, so plate versus bar is **not** a controlled comparison.
- 6 and 10 hidden frames; every interval is wide. Hand labels behind the plate are my best guess, with several degrees of
  uncertainty. The 2D labels cannot judge 3D.
- No claim of generalisation. The plate is also not the only difference between the clips, so I call it a lead, not a cause.

## Next steps

1. **Held-out check, same camera.** Film a plate set with the knee hidden and a set with the knee visible (empty bar or small plates),
   same position and settings, label 15 to 20 frames each, and run the existing scripts unchanged. This removes the camera confound
   and tests whether the findings and the detectors hold up on data they were not built on.
2. **A real reference for the hidden knee.** A second synchronised camera, or a dataset with 3D ground truth (Fit3D, access pending), to
   replace my guesses behind the plate.
3. **Training-time fixes.** Fine-tune a pose model on synthetic plate occluders and test it on real frames, the direction the literature
   suggests. This needs far more labelled data than I have.

## Repository

| Path | Content |
|---|---|
| `src/pose/` | pose extraction (MediaPipe, RTMPose), point tracker, export for the 3D lifter |
| `src/metrics/` | angles and rep logic, labelling tool, evaluation scripts (`eval_gap.py` is the main metric) |
| `labels/` | my hand labels (CSV); `labels/strict/` is a pass with a written rule |
| `notebooks/` | Colab notebook for the 2D-to-3D lifter |
| `docs/` | [RESULTS.md](docs/RESULTS.md) (all results in detail), [SCRIPTS.md](docs/SCRIPTS.md) (what each script does), figures |

To reproduce the headline figure (needs your own clip in `data/raw/`; footage is not included because it shows a person and a gym):

```
python src/pose/extract_mediapipe.py data/raw/<clip>.mp4
python src/pose/extract_rtmpose.py data/raw/<clip>.mp4 --mode balanced
python src/pose/export_h36m.py <clip>
python src/metrics/eval_gap.py <clip>
python src/metrics/plot_gap.py
```

## Related work

Stanford CS231n 2024, *Automating powerlifting judging through keypoint detection* (pretrained keypoints plus rules).
Occlusion-robust pose estimation: [DAG](https://arxiv.org/abs/2401.00155), [BlanketGen2-Fit3D](https://arxiv.org/abs/2501.12318),
[SAM-Body4D](https://arxiv.org/abs/2512.08406), [4DHumans](https://openaccess.thecvf.com/content/ICCV2023/html/Goel_Humans_in_4D_Reconstructing_and_Tracking_Humans_with_Transformers_ICCV_2023_paper.html).

## Privacy

Raw video and generated overlays are excluded (`.gitignore`) because they show a person and a gym.
