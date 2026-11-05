# Phase 3 — Playing with the controller (in progress, 27-10-2026)

**Goal:** connect the Phase 2 logic to the game: real keys, a complete loop and measured latency.

## Done (tested without a camera, with videos and a simulated keyboard; 81 tests)

| Module | What it does |
|---|---|
| `air_controller/output.py` | Layer 5. Translates button events into keys with pynput. Remembers which keys it has pressed and always releases them: on exit, on pause, on recalibration, on an error or with Ctrl+C. Validates key names at startup. `RecordingKeySender` for tests and `--dry-run`. |
| `air_controller/latency.py` | Per-stage and end-to-end latency (frame received → key sent), p50/p95/max, and loop FPS. |
| `air_controller/capture.py` | Each frame stores the instant it arrived from the camera. |
| `air_controller/dashboard.py` | Everything on screen: guided calibration with centered instructions and SUCCESS, and the game dashboard (both hand cards plus input table). The window is centered and scaled to fit the screen. |
| `scripts/play.py` | The complete controller: calibrate → play. P pauses, R recalibrates, Q quits. `--dry-run` prints the keys instead of sending them. Prints the latency on exit. |

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

## Pending

1. **Accessibility permission** on macOS for the app it runs from (VS Code or Terminal). Without it, pynput doesn't raise an error but the keys don't arrive.
2. **Confirm FullScreenMario's jump key.** According to the documentation found, you move with the arrows or WASD and Shift is for running; the jump is not confirmed. `config.toml` has a provisional `a = "up"`.
3. **Measure real FPS and latency** with `play.py --dry-run` and then while playing. If the loop stays at ~15 fps, optimize: draw less, or move MediaPipe to its own thread / LIVE_STREAM mode.
4. ~~**Button B**~~ Done: tilting the right thumb outwards. Key profiles per game are selected with `play.py --game <name>`, e.g. `celeste`.
