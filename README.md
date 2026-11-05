# Air Gesture Gaming Controller

Play keyboard games with an invisible NES controller. A regular webcam and [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) track your thumbs, and your gestures become real key presses in whichever window has focus.

## Quick start

You need Python 3.11 or newer and a webcam. It has been developed and measured on a MacBook Pro M1.

```bash
git clone https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System.git
cd Air-Gesture-Gaming-Control-System

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# The hand model is not stored in the repo: download it once
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Try it without sending any keys, then play for real:

```bash
python scripts/play.py --dry-run         # prints the keys instead of sending them
python scripts/play.py                   # default keys
python scripts/play.py --game celeste    # keys from the [games.celeste] profile
```

**On macOS**, give the app you run this from (Terminal or VS Code) the Accessibility permission in System Settings. Without it there is no error, but the keys never reach the game.

## How to play

1. **Calibrate.** Hold both hands up in vertical fists, thumbs up, as if holding a controller. The screen guides you through five poses, and calibration runs every time you start because references from one session don't carry over to another.
2. **Click the game window.** Keys go to whichever window has focus.
3. **Play.**

| Button | Gesture | Default key | `--game celeste` |
|---|---|---|---|
| Left | Tilt the left thumb outwards | `left` | `left` |
| Right | Tilt the left thumb inwards, towards the other hand | `right` | `right` |
| A | Push the right thumb down | `up` | `c` |
| B | Tilt the right thumb outwards | `down` | `x` |

A and B use the same thumb, so they can't be pressed at the same time.

| Key (click the controller window first) | Action |
|---|---|
| `P` | Pause or resume; releases all keys |
| `R` | Recalibrate; releases all keys |
| `Q` | Quit and print the latency summary |

**Emergency brake:** move your hands out of view. All keys are released after 200 ms.

## Configuration

Thresholds, timings and keys are all in [config.toml](config.toml). The file is validated at startup, so a typo or an out-of-range value stops the program with a message saying what to fix.

To add a game, add a profile that lists only the keys that differ from `[keys]`, then run it with `--game mygame`:

```toml
[games.mygame]
a = "space"
b = "z"
```

## Getting every update

All work goes to the `main` branch, and each push is one or more commits you can open or download. The repository is private, so you need to be added as a collaborator and signed in to GitHub.

**Get the latest version**

```bash
git switch main
git pull
pip install -r requirements.txt    # in case dependencies changed
```

**See what's new before updating**

```bash
git fetch
git log --stat HEAD..origin/main
```

**Go to any earlier version**

```bash
git log --oneline                  # list all versions
git switch --detach <commit>       # go to one, e.g. 27c0768
git switch main                    # come back to the latest
```

**Without using git**

| What you want | Where |
|---|---|
| Every push, newest first | [Commit history](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commits/main) |
| Browse the files of one version | `https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/tree/<commit>` |
| Download one version as a ZIP | `https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/archive/<commit>.zip` |

### Main milestones

The commit history is the complete list. Versions older than `ff1a4eb` have Spanish text and may need different setup steps.

| Date | Commit | What changed |
|---|---|---|
| 2026-11-05 | [`ff1a4eb`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/ff1a4eb) | Merged B button, game profiles, English text and dashboard |
| 2026-11-05 | [`bd8c6a7`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/bd8c6a7) | B button: right thumb outwards (dash in Celeste) |
| 2026-11-05 | [`727f9ae`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/727f9ae) | Key profiles per game (`--game`) |
| 2026-11-05 | [`5fc70d4`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/5fc70d4) | English translation and game dashboard |
| 2026-10-27 | [`27c0768`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/27c0768) | First latency measurement on M1 |
| 2026-10-27 | [`7bdedce`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/7bdedce) | Phase 2 closed; key sending and the complete controller |
| 2026-10-26 | [`e36ab82`](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commit/e36ab82) | Spike: results of the second session |

## Project status

The controller works end to end, and work is in Phase 3 (playing real games).

| Measurement | Result |
|---|---|
| Accuracy in a 2.5-minute guided session, worst button | 99.4 % (target ≥ 95 %) |
| Exact A taps in that session | 9 of 9 |
| Frame to key, p50 / p95 | 49 / 56 ms |
| Estimated delay you feel, including press confirmation and camera | ~130–160 ms |
| Loop speed | 19.6 fps |

These results were measured on 27 October 2026, before button B was added. The guided session doesn't ask for B; it only counts accidental B presses.

Open items are listed in [docs/phase3-progress.md](docs/phase3-progress.md).

## How it works

```
camera → MediaPipe hands → classifier → state machine → keys
                               ↑
                        your calibration
```

| Where | What it does |
|---|---|
| [air_controller/capture.py](air_controller/capture.py) | Always reads the newest camera frame, so delay doesn't build up |
| [air_controller/calibration.py](air_controller/calibration.py) | Learns your five poses and rejects a poor calibration with a reason |
| [air_controller/classifier.py](air_controller/classifier.py) | Decides which buttons are pressed in each frame |
| [air_controller/state_machine.py](air_controller/state_machine.py) | Filters flicker, then confirms presses and releases |
| [air_controller/posture.py](air_controller/posture.py) | Warns if a hand rotates or moves too far from its calibrated size |
| [air_controller/output.py](air_controller/output.py) | Sends keys and makes sure none stay stuck |
| [air_controller/dashboard.py](air_controller/dashboard.py) | Draws the calibration screens and game dashboard |
| [scripts/play.py](scripts/play.py) | The complete controller |
| [scripts/calibrate_live.py](scripts/calibrate_live.py) | Calibrates and shows the live reading without sending keys |
| [scripts/guided_session.py](scripts/guided_session.py) | Runs the 2.5-minute accuracy test; score it with `scripts/evaluate_guided.py` |

MediaPipe runs on the CPU on purpose: on the M1 the GPU was about 4 times faster but lost hands and duplicated detections. The reasoning behind this and other decisions is in the docs:

- [Spike results](docs/spike-results.md)
- [Phase 2 progress](docs/phase2-progress.md)
- [Phase 3 progress](docs/phase3-progress.md)

## Running the tests

The tests need no camera:

```bash
python -m pytest tests
```

Run them from the activated `.venv`, and point pytest at `tests`, because the `scripts/*_test.py` files open the camera.
