# Phase 3 — Playing with the controller (in progress, updated 05-11-2026)

**Goal:** connect the Phase 2 logic to the game: real keys, a complete loop and measured latency.

## Done (tested without a camera, with videos and a simulated keyboard; 153 tests, run on every push)

| Module | What it does |
|---|---|
| `air_controller/output.py` | Layer 5. Translates button events into keys with pynput. Remembers which keys it has pressed and always releases them: on exit, on pause, on recalibration, on an error or with Ctrl+C. Validates key names at startup. `RecordingKeySender` for tests and `--dry-run`. |
| `air_controller/latency.py` | Per-stage and end-to-end latency (frame received → key sent), p50/p95/max, and loop FPS. |
| `air_controller/capture.py` | Each frame stores the instant it arrived from the camera. |
| `air_controller/dashboard.py` | Everything on screen: guided calibration with centered instructions and SUCCESS, and the game dashboard (both hand cards plus input table). The window is centered and scaled to fit the screen. |
| `scripts/play.py` | The complete controller: calibrate → play. P pauses, R recalibrates, Q quits. `--dry-run` prints the keys instead of sending them. Prints the latency on exit. |
| `config.toml` `[games.<name>]` | Key profiles per game, selected with `play.py --game <name>`. Only the keys that differ from `[keys]` are listed. |
| Button B | Right thumb tilted outwards. Calibration has a fifth pose and rejects B if it looks like rest or like A, or if the left thumb moves while pressing it. |
| `air_controller/runtime.py` | The game loop in its own thread: frame → hands → classifier → state machine → keys. The window draws the latest result in parallel, so drawing never delays a key. |
| `air_controller/play_log.py`, `scripts/analyze_play.py` | Every game is recorded in `data/play/` (landmarks, readings, keys) and can be diagnosed afterwards: lost hands, flicker, A/B confusion, slow frames. |
| `air_controller/rest_tracker.py` | Re-learns where the right thumb rests while you play (A and B), from the last 5 s. |
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

## First game with the dashboard (Celeste, 05-11-2026)

| Stage | 27-10 (dino, old screen) | 05-11 (Celeste, new dashboard) |
|---|---|---|
| Frame → key, p50 / p95 | 49 / 56 ms | 61 / 77 ms |
| Detection (MediaPipe), p50 / p95 | 37 / 37 ms | 42 / 97 ms (max 427) |
| Loop FPS | 19.6 | 13.0 |

- Calibration accepted on the first try with B: left thumb moved 5 % while pressing A and 8 % while pressing B (limit 15 %).
- **The loop slowed down to 13 fps** (~77 ms per frame): ~42 ms of MediaPipe plus ~35 ms of drawing and showing the larger dashboard. Because a press needs two frames in a row, the drawing time was added to every press: estimated perceived delay ~170–200 ms.
- **MediaPipe got slower and jumpier** (p95 97 ms), probably because the browser running the game competes for the CPU.

**Decision: the key path runs in its own thread** (`runtime.py`); the window draws in parallel. On the spike's `combo` video, with the dashboard drawn, the loop went from 43 to 55 fps on a cloud CPU with the same presses (30 jumps) and no stuck keys. On the M1 the gain should be bigger (drawing cost ~35 ms there): expected ~20+ fps, limited by MediaPipe. To be measured.

## False presses in Celeste: the resting thumb drifts (05-11-2026)

With the threaded loop: 19.9 fps, MediaPipe p50 54 / p95 60 ms (the spikes were gone after making OpenCV draw on one thread), frame → key 68 / 85 ms. But the player felt presses he hadn't made. The recorded game (96 s) showed why:

| Window (s) | 0–15 | 15–30 | 30–45 | 45–60 | 60–75 | 75–90 | 90–105 |
|---|---|---|---|---|---|---|---|
| Relaxed right thumb (0 = calibrated rest, 1 = A pressed) | 0.19 | 0.42 | 0.33 | 0.39 | 0.57 | 0.58 | 0.67 |
| A "held" | 45 % | 56 % | 60 % | 53 % | 87 % | 94 % | 61 % |

- The right fist rolled from −4° to −20° during the game, and the thumb is measured on screen axes, so rolling looks like the thumb going down. Also, a relaxed thumb sits lower than the straight thumb held while calibrating.
- **Hysteresis (separate press and release thresholds) was tested and rejected:** with the rest already past the release point, A stayed stuck even longer (99 % in one window).
- **Decision: re-learn the rest while playing** (20th percentile of the last 5 s, used once there are 4 s of history). A first version (rest capped at halfway, progress capped at 1) fixed the first game but not a second one (23:15, no fist roll): there the relaxed thumb sat at 0.6–1.0 of the calibrated press, while real presses went deeper, up to 1.2–2.8. Measuring the thumb as a projection on the rest → press axis (it can go past 1) and letting the rest move up to 0.8, a button is pressed 0.4 past the current rest. Replayed:

  | | A held per 15 s, before | after |
  |---|---|---|
  | Game 1 (23:03) | 57–87 % | 23–44 % |
  | Game 2 (23:15) | 5–100 % | 9–47 % |
  | Guided session 27-10 | worst button 99.4 %, 9/9 A taps | the same |
  | Spike clips | 0 / 30 / 29 presses | the same (±1) |

  **Limit:** a button held for more than ~4 s is released (in the spike's 29 s `a_held` clip, A was held 14 % of the clip instead of 100 %). Fine for Celeste and the dino; a game where you hold the accelerator needs `rest_window_ms = 0` in its `[games.<name>]` profile.
- **Button B didn't work well** (the player's report, confirmed in both games). Tilting the thumb outwards also lowers its tip: in the calibrations the B pose was 60–97 % of the way to A. With "the clearer button wins", A and B were close to a tie, and in the second game B's re-learned rest rose to ~0.5: 28 % of the outward tilts were read as A or nothing. **Decision: B wins whenever the thumb is outwards** (pressing A moves it clearly inwards: −1.4 and −2.0 along B), **and only A's rest is re-learned.** Replayed: 100 % of the tilts read as B in both games, A unchanged. Calibration now rejects an A press that drifts outwards (≥ 20 % of the way to B), which would otherwise become B.
- **B still had false presses** (third game, 23:29: 28 B sent). Most were the relaxed thumb wandering outwards *without* going down (outwards 0.4–0.9, down ≤ 0), while the player's calibrated B is outwards *and* down (0.6–0.97 of the way to A). **Decision: B needs both**, 0.6 of the outward tilt and 0.4 of the calibrated lowering (from the calibrated rest; a B that doesn't lower the thumb only needs the tilt). Replayed with the real code: game 3 28 → 11 B presses with all 11 moments in the calibrated B shape caught; game 2 7 → 4 (4/4); game 1 24 → 17 (13/14). A unchanged; guided session unchanged.
- **Strong dash attempts were lost** (06-11, 11:17, the best game so far: 36 clean jumps, D-pad near its threshold only 6.7 % of the time). That calibration's B lowered the thumb as much as A, so "down 0.4 of B's" asked for a lot, and the player confirmed dashing at 11–18 s and 41–44 s with strong outward tilts (1.3–2.0) that didn't go down: none was sent. **Decision: a tilt of 1.2 or more counts on its own.** Replayed: those attempts now send B (8 → 13 in that game), and the earlier games gain 1–3 B presses (game 3: 11 → 14). Only one strong tilt is still unread.
- Also seen in that game: the D-pad spent 34 % of the time within ±0.1 of its threshold and the right hand was lost 4 times (once for 2.7 s).

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

1. **Play Celeste again** with the re-learned rest and compare with `analyze_play.py`: A held % should stay flat over the game.
2. **Dash up and diagonals** need an "up" direction on the D-pad, which the left thumb can't give: probably raising the left fist. Only worth it if Celeste is playable with horizontal dashes.
3. **Add B to the guided session** so its accuracy is measured, not only its accidental presses. Old sessions must still be scored with the old script.
4. **Measure the threaded loop on the M1 while playing Celeste.** If MediaPipe still spikes (p95 ≫ p50), check what else is using the CPU; the browser running the game is the first suspect.
5. The calibration screens and the guided session still run in series (detect, then draw). That's fine: no keys are sent there.
