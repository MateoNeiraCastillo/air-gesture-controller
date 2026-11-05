# Air Gesture Gaming Controller

Play keyboard games with an invisible NES controller. A regular webcam and MediaPipe track your thumbs, and your gestures become real key presses in whichever window has focus.

## Setup

You need Python 3.11 or newer and a webcam.

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

**On macOS**, give the app you run this from (Terminal or VS Code) the Accessibility permission in System Settings. Without it there is no error, but the keys never reach the game.

## Run

```bash
python scripts/play.py --dry-run         # prints the keys instead of sending them
python scripts/play.py                   # default keys
python scripts/play.py --game celeste    # keys from the [games.celeste] profile
```

## How to play

1. **Calibrate.** Hold both hands up in vertical fists, thumbs up, as if holding a controller. The screen guides you through five poses each time you start.
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
| `Q` | Quit |

To release all keys quickly, move your hands out of view.

## Add a game

Keys and thresholds are in [config.toml](config.toml). Add a profile that lists only the keys that differ from `[keys]`, then run it with `--game mygame`:

```toml
[games.mygame]
a = "space"
b = "z"
```

## Get updates

All work goes to the `main` branch. The repository is private, so you need to be added as a collaborator and signed in to GitHub.

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

**Without git:** every push is listed in the [commit history](https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/commits/main), where you can browse the files of any version. To download one as a ZIP, use `https://github.com/Parth-fintech/Air-Gesture-Gaming-Control-System/archive/<commit>.zip`.

## Tests

From the activated `.venv`, no camera needed:

```bash
python -m pytest tests
```

## More

Results and design decisions are in [docs/](docs/).
