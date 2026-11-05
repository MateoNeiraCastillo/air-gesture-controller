# Phase 2 — Controller logic (CLOSED on 27-10-2026)

**Goal:** turn what was learned in the spike into the controller's logic: from per-frame landmarks to reliable presses, testable without a camera. Integration with the game (keys, FullScreenMario) is Phase 3.

## Done

| Step | Module | What it does |
|---|---|---|
| 2.1 | `buttons.py`, `config.py`, `config.toml` | LEFT/RIGHT/A buttons; all configuration in TOML, validated at startup (a typo or an out-of-range value prevents startup) |
| 2.2 | `calibration.py` | 4 poses (rest, A pressed, D-pad left, D-pad right) with valid frames, medians and quality control |
| 2.3 | `classifier.py` | Button A: nearest centroid (thumb height). D-pad: direction = nearest calibrated target; pressed if it covers ≥ 30 % of the travel from rest |
| 2.4 | `posture.py` | Warns if a hand leaves the range of sizes seen during calibration for > 0.5 s (hand rotated or at a different distance) |
| 2.5 | `state_machine.py` | Press/release after 40 ms in a row (started at 80; see result); no hand = no information (holds); > 200 ms without a hand → releases its buttons |
| — | `capture.py` | Camera in a thread that keeps the most recent frame (MediaPipe takes ~38 ms and the camera delivers one every ~33 ms) |
| — | `scripts/calibrate_live.py` | Live calibration + view of what the game would receive |
| 2.6 | `guided.py`, `scripts/guided_session.py`, `scripts/evaluate_guided.py` | ~2.5 min guided session (3 identical blocks) and its evaluation |

69 tests pass at the close of the phase.

## Decisions and findings

1. **The left thumb moved when pressing A.** In the spike recordings, while A was pressed, the left thumb moved to where "right" had been calibrated (night: 76–110 % of the travel). In the game, Mario would have walked right when jumping. No threshold fixes it: the "right" gesture was too small and in the same direction as "left".
   - **Calibration:** in the "A pressed" pose the left hand is also checked; if it moves towards a direction (≥ 15 % of the travel), the calibration is rejected with a message saying what to repeat.
   - **New gesture** (vertical fists, thumb up): rest = thumb straight; left = tilt it outwards; right = inwards, towards the other hand; when pressing A, the left thumb stays straight.
   - **Live result (2 calibrations):** accepted on the first try; right–rest separation 17 (previously 7–14), left–right 23–44; left thumb when pressing A: 0 % towards right (previously 50–52 %).
2. **D-pad threshold: 0.3 of the travel** (not 0.5). In the daytime `combo` the thumb covered ~50 % of the calibrated travel; with 0.5 it got 49 % right, with 0.3 98 %. At rest it rarely goes past 0.2.
3. **The D-pad direction is chosen by the nearest target, not by the largest progress.** In the night session left and right pointed the same way (cosine +0.95) and "right" was a short axis: a "left" point progressed 3 times further along that axis.
4. **The posture monitor compares against a range, not a single value.** The hand tilts when the thumb moves (the left one ranged from 77 % to 130 % of its median depending on the pose, all valid). With a single value, false warnings fired.
5. **State machine on the spike:** night `rhythm` 32 raw presses → 30 confirmed; `combo` 29–30 → 29. In the daytime `rhythm` (rotated fist) it confirms none, but the monitor warns 76 % of the time: it fails with a warning, not silently.

## Guided session result (27-10-2026)

| Criterion | Result | Target |
|---|---|---|
| Worst button (state the game receives vs what was asked) | **99.4 %** (left 99.4, right 100, A 99.7) | ≥ 95 % |
| Exact A taps | **9 / 9** | ≥ 80 % |
| Drift block 1 → 3 | none (99.1 → 100 → 100 %); left thumb at rest between −4 % and +8 % of the travel | — |
| Posture warnings | 0 % | — |

- **The only failure** (a "Rest" in block 1 at 73 % on the D-pad) was human: after "Hold RIGHT", with "Next: Hold LEFT" on screen, the thumb went all the way left (100–127 % of the travel) for ~0.45 s and came back. The system correctly read a real gesture; it happened once, in the first block.
- **The session loop ran at ~15 fps** (66 ms between frames), not 30: MediaPipe ~38 ms + drawing + saving data.
- **Press confirmation** (session re-evaluated offline with different values): 0 ms → exact taps 78 % (ghosts); 40/60/80 ms → 100 %, with ~100 ms of delay in every case, because it is set by the frame rate. **Decision: 40 ms** (same accuracy; less delay once the loop is faster).

## Conclusion

The invisible controller's logic works in continuous use for ~2.5 min without recalibrating. **The bottleneck is now the frame rate**, which is the first thing to measure and improve in Phase 3.
