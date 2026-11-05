# Spike: invisible NES controller

**Question:** can a regular webcam + MediaPipe read an imaginary NES controller held with both hands: D-pad with the left thumb and button A with the right?

**Answer:** yes within a session, with calibration at startup and CPU (confirmed in two sessions). Open risk for Phase 2: if the hand rotates during play, the measurements change.

## Setup

- MacBook Pro M1, built-in webcam, 640×480 at 30 fps, artificial light.
- MediaPipe 1.1.1, `HandLandmarker` in VIDEO mode, `num_hands=2`, thresholds 0.5.
- 6 clips of 30 s, each with a single held pose: `rest`, `a_held`, `a_rhythm`, `dpad_left`, `dpad_right`, `combo` (right held + A in rhythm).
- Pipeline: `record.py` (raw video) → `extract.py` (landmarks to `.npz`) → `analyze.py` (metrics and plots).

## Results

| Metric | Result | Target |
|---|---|---|
| Frames with two hands (CPU) | 98.0 % | ≥ 95 % |
| Button A, rest vs pressed (calibrated) | 100 % | ≥ 95 % |
| `a_rhythm`: presses detected | 30 in 29 s | ~1/s |
| `combo`: A presses detected | 29 in 29 s | ~1/s |
| D-pad, rest / left / right (calibrated) | 99.1 % | ≥ 95 % |
| `combo`: D-pad read as "right" | 100 % | ≥ 95 % |
| CPU inference on M1 (p50 / p95) | 38 / 39 ms | ≤ 35 ms |

Calibration: the first 90 valid frames (~3 s) of each pose place the centroids of a centroid classifier; the rest of the clip is used to evaluate. `rhythm` and `combo` are not used to calibrate anything: they are the out-of-sample test.

## What we learned (and what went wrong)

1. **The GPU ruins detection.** Confirmed on the M1 with `compare_delegates.py`, on the same videos:

   | Clip | CPU: 2 hands | GPU: 2 hands | GPU: duplicated | CPU p50 | GPU p50 |
   |---|---|---|---|---|---|
   | `rest` | 100 % | 6 % | 13 % | 38 ms | 12 ms |
   | `a_held` | 100 % | 39 % | 0 % | 38 ms | 10 ms |
   | `dpad_right` | 88 % | 76 % | 0 % | 38 ms | 7 ms |

   The GPU is ~4 times faster but unusable. **Decision: CPU.** It should be retested with future MediaPipe versions.
2. **Thumb angles don't work; position does.** With the thumb bent over the fist, MediaPipe misestimates the joint: there were clearly pressed frames with the rest angle. The tip position relative to the index finger base and the wrist, normalized by hand size, does separate.
3. **For button A only the tip height matters.** The horizontal components change if the hand rotates slightly between clips, and led the classifier to detect 0 presses in `rhythm`. Using only the vertical ones: 30 out of ~29.
4. **In this posture, the D-pad shows up mostly vertically.** With the fist vertical and the thumb up, "left" and "right" move the tip mostly vertically. With dx alone, right vs rest gave 55 %; with all 4 features, 99 %.
5. **Calibration has to wait until it sees the hand.** In `dpad_right`, the left hand wasn't detected for the first few seconds. Calibrating with "the first 3 s" left the class empty, so calibration uses the first N valid frames.

## Session 2 (day, natural light)

| Metric | Night | Day |
|---|---|---|
| Two hands (CPU) | 98 % | 100 % |
| Button A calibrated | 100 % | 99.9 % |
| `rhythm`: presses | 30 / 29 s | 23 / 29 s |
| D-pad calibrated | 99.1 % | 99.9 % |
| `combo`: D-pad "right" | 100 % | 81 % |

**Across sessions, without recalibrating:** button 53–62 %, D-pad 62–67 %, close to chance. **Decision: calibration is mandatory every time you start playing.**

The two worst daytime numbers have an identified cause:

- **`rhythm` (23/29):** in that clip the fist was rotated to face the camera. The scale (wrist → middle finger base) projects almost to a point (1–29 px instead of ~60) and the features blow up.
- **`combo` (81 %):** when doing two things at once, the D-pad "right" is pushed less than in `dpad_right` and ends up between rest and right.

We tried other scales (knuckle width, maximum of both, pixels) and MediaPipe's 3D coordinates in camera and hand axes. None worked in both sessions at once: they fixed one and broke the other. We kept the original scale, which is the best overall.

## Limitations

- Each clip is an isolated 30 s pose. We don't know how much the posture drifts over several minutes of continuous play.
- `dpad_right` had 12 % of frames without the left hand on CPU.

## Next

1. Design Phase 2 for drift: calibration at startup; D-pad as a direction relative to rest with a dead zone (not "the nearest centroid"); visual warning if the hand rotates (the scale drops relative to the calibration); a key to recalibrate.
1b. Realistic experiment: a continuous 2–3 min recording mixing actions, to measure drift within a session.
2. Latency: at ~38 ms per frame, the loop doesn't reach the camera's 30 fps. Capture must always keep the most recent frame (a capture thread that discards old ones); otherwise frames pile up in the buffer and latency grows. Measure end-to-end latency with the game running (Phase 3).
