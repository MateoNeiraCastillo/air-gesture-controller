# Phase 3 — Playing with the controller (in progress, updated 05-11-2026)

**Goal:** connect the Phase 2 logic to the game: real keys, a complete loop and measured latency.

## Done (tested without a camera, with videos and a simulated keyboard; 123 tests)

| Module | What it does |
|---|---|
| `air_controller/output.py` | Layer 5. Translates button events into keys with pynput. Remembers which keys it has pressed and always releases them: on exit, on pause, on recalibration, on an error or with Ctrl+C. Validates key names at startup. `RecordingKeySender` for tests and `--dry-run`. |
| `air_controller/latency.py` | Per-stage and end-to-end latency (frame received → key sent), p50/p95/max, and loop FPS. |
| `air_controller/capture.py` | Each frame stores the instant it arrived from the camera. |
| `air_controller/dashboard.py` | Everything on screen: guided calibration with centered instructions and SUCCESS, and the game dashboard (both hand cards plus input table). The window is centered and scaled to fit the screen. |
| `scripts/play.py` | The complete controller: calibrate → play. P pauses, R recalibrates, Q quits. `--dry-run` prints the keys instead of sending them. Prints the latency on exit. |
| `config.toml` `[games.<name>]` | Key profiles per game, selected with `play.py --game <name>`. Only the keys that differ from `[keys]` are listed. |
| Button B | Right thumb tilted outwards. Calibration has a fifth pose and rejects B if it looks like rest or like A, or if the left thumb moves while pressing it. |
| `air_controller/recordings.py` | Reads the recordings made before the translation (`reposo_noche_01.npz`, `sesion_*.npz`) under their English names, so the analysis scripts still work on them. |

**Emergency brake:** moving your hands away from the camera releases all keys after 200 ms (`missing_hand_ms`).

## First real measurement (M1, `play.py --dry-run`, 27-10-2026)

| Stage | p50 | p95 |
|---|---|---|
| Frame wait (age when picked up) | 18 ms | 33 ms |
| Detection (MediaPipe, CPU) | 37 ms | 37 ms |
| Classify + state machine | 0.1 ms | 0.1 ms |
| **Frame → key** | **49 ms** | **56 ms** |
| Loop FPS | 19.6 | |

- Always reading the most recent frame works: the average wait is half a frame (18 ms) and doesn't grow.
- Very stable: p95 almost equal to p50.
- **Does not include** the press confirmation (2 consecutive frames: +~50 ms at 19.6 fps) or the camera's internal delay (~30–60 ms typical). Estimated perceived delay: **~130–160 ms**, at the edge of the 100–120 ms budget.
- The loop runs at 19.6 fps (~51 ms per iteration): 37 ms of MediaPipe + ~14 ms of drawing and window. First optimization if lag is noticeable while playing: make drawing cheaper (cheaper than touching MediaPipe).

## Games

| Game | Keys | Status |
|---|---|---|
| Chrome dino (`chrome://dino`) | default (`up` jumps, `down` ducks) | Played: works well |
| [Celeste Classic](https://maddymakesgamesinc.itch.io/celesteclassic) (PICO-8, browser) | `--game celeste` (`c` jumps, `x` dashes) | Next |
| FullScreenMario | — | Dropped: the project was taken down after DMCA notices from Nintendo |

## Decisions and findings

1. **Each right-thumb button is judged on its own axis** (A on the thumb's height, B on its sideways position). The first version of B decided both buttons on all four thumb features. On the 27-10 guided session that scored the same as before, but replaying the spike showed A stuck on in the night clips (`a_rhythm`: 0 presses instead of 30): the thumb's sideways position drifted between calibration and play, and since sideways noise is tiny, that drift alone outweighed the height. With one axis per button, A is back to 30 of ~32 raw presses.
2. **Resuming after a pause starts from "nothing pressed"**, so a button held at that moment is confirmed and sent again. Before, its press happened while paused and never reached the game.

## Pending

1. **Play Celeste** with `--game celeste` and note what fails (late jumps, wrong direction, accidental B).
2. **Dash up and diagonals** need an "up" direction on the D-pad, which the left thumb can't give: probably raising the left fist. Only worth it if Celeste is playable with horizontal dashes.
3. **Add B to the guided session** so its accuracy is measured, not only its accidental presses. Old sessions must still be scored with the old script.
4. **Measure FPS with the new dashboard.** It draws a larger canvas (~7 ms per frame on a cloud CPU, after the keys are sent, so it doesn't delay them, but it lowers the FPS). If the loop drops below ~18 fps, draw the static parts once.
