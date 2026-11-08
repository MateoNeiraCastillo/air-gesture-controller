# Air Gesture Gaming Controller

[![tests](https://github.com/MateoNeiraCastillo/air-gesture-controller/actions/workflows/tests.yml/badge.svg)](https://github.com/MateoNeiraCastillo/air-gesture-controller/actions/workflows/tests.yml)

![Playing Subway Surfers with thumb gestures](docs/demo.webp)

Play keyboard games with an invisible NES controller. A regular webcam and MediaPipe track your thumbs, and your gestures become real key presses in whichever window has focus.

Made for [Subway Surfers](https://poki.com/en/g/subway-surfers) in the browser: switch lanes with the left thumb, jump and roll with the right one.

## Results

Measured on a MacBook Pro M1 with its built-in webcam.

| Measurement | Result |
|---|---|
| 2.5-minute guided session: worst button / exact A taps | 99.4 % / 9 of 9 |
| Playing Subway Surfers (1 min): presses sent, hands in view | 39 (16 lane changes, 14 jumps, 9 rolls), 98–99 % |
| Playing Subway Surfers: loop speed, MediaPipe per frame | 12 fps, 76 ms (the 3D game shares the computer with the controller) |
| Frame → key (p50 / p95), in a lighter 2D game | 68 / 85 ms |
| Relaxed thumb read as "A held", worst 15 s of each game (2 games, replayed) | 94 % and 81 % → 42 % and 47 % |
| Accidental B presses in one game (replayed) | 28 → 14, every real B still caught (11 of 11) |
| Tests, run on every push (Python 3.11–3.13) | 153 |

How each number was measured, and what changed to get it, is in [docs/phase3-progress.md](docs/phase3-progress.md).

## Setup

You need Python 3.11 or newer and a webcam.

```bash
git clone https://github.com/MateoNeiraCastillo/air-gesture-controller.git
cd air-gesture-controller

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# The hand model is not stored in the repo: download it once
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**On macOS**, give the app you run this from (Terminal or VS Code) the Accessibility permission in System Settings. Without it there is no error, but the keys never reach the game.

## Run

```bash
python scripts/play.py --dry-run         # prints the keys instead of sending them
python scripts/play.py                   # plays: the default keys are Subway Surfers'
python scripts/analyze_play.py           # what went wrong in the last game
```

Every game is saved in `data/play/` so it can be analyzed afterwards. Add `--no-record` to `play.py` to skip that.

## How to play

1. **Open the game.** [Subway Surfers on Poki](https://poki.com/en/g/subway-surfers), on one side of the screen, leaving room for the controller window.
2. **Calibrate.** Hold both hands up in vertical fists, thumbs up, as if holding a controller, with your thumbs relaxed as you'll have them while playing. The screen guides you through five poses each time you start.
3. **Click the game window.** Keys go to whichever window has focus.
4. **Play.**

| Button | Gesture | Key | In Subway Surfers |
|---|---|---|---|
| Left | Tilt the left thumb outwards | `left` | Move one lane left |
| Right | Tilt the left thumb inwards, towards the other hand | `right` | Move one lane right |
| A | Push the right thumb down | `up` | Jump |
| B | Tilt the right thumb outwards, the same way as during calibration | `down` | Roll |

Every move in Subway Surfers is a tap: move the thumb and bring it back. A and B use the same thumb, so they can't be pressed at the same time.

| Key (click the controller window first) | Action |
|---|---|
| `P` | Pause or resume; releases all keys |
| `R` | Recalibrate; releases all keys |
| `Q` | Quit |

To release all keys quickly, move your hands out of view.

## When something feels wrong

Run `python scripts/analyze_play.py` after a game. It reads the saved game and reports, in plain words:

- how often each hand was lost, and for how long;
- presses the camera saw but that were too short to send, and holds that were cut;
- how long the D-pad sat near its threshold, where it flickers;
- how often A and B were confused;
- frame rate and stutters.

It ends with suggestions, such as more light or a different calibration.

## Other games

Any keyboard game that uses left, right, up and down works with the default keys: the Chrome dino (`chrome://dino`) or [Jstris](https://jstris.jezevec10.me) (Tetris), for example.

For a game with other keys, add a profile to [config.toml](config.toml) that lists only the keys that differ from `[keys]`, then run it with `--game mygame`. A `celeste` profile is included as an example.

```toml
[games.mygame]
a = "space"
b = "z"
```

While you play, the controller re-learns where your right thumb rests, so a thumb that relaxes during the game doesn't press A on its own. As a side effect, A held for more than about 4 seconds is released. If your game needs long holds, such as an accelerator, turn it off in the profile with `rest_window_ms = 0`.

## Get updates

All work goes to the `main` branch.

**Latest version**

```bash
git switch main
git pull
pip install -r requirements.txt    # in case dependencies changed
```

**Any earlier version**

```bash
git log --oneline                  # list all versions
git switch --detach <commit>       # go to one, e.g. 27c0768
git switch main                    # come back to the latest
```

**Without git:** every push is listed in the [commit history](https://github.com/MateoNeiraCastillo/air-gesture-controller/commits/main), where you can browse the files of any version. To download one as a ZIP, use `https://github.com/MateoNeiraCastillo/air-gesture-controller/archive/<commit>.zip`.

## Tests

From the activated `.venv`, no camera needed:

```bash
python -m pytest tests
```

## More

Results and design decisions are in [docs/](docs/).

## Authors

Mateo Neira Castillo and Parth Deshmukh.
