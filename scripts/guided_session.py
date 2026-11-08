"""~2.5-minute guided session: the Phase 2 exam, or data collection for the ML models.

1. You calibrate as in calibrate_live.py (R to repeat if rejected).
2. For ~2.5 min the screen asks you for actions (hold right, press A
   3 times, right + A...). The current one is at the top and the next below it.
   The bottom panel shows what the game would receive, as in the live view.
3. When it finishes (or on Q) everything is saved in data/guided/ for analysis with:
       python scripts/evaluate_guided.py

Usage:
    python scripts/guided_session.py                              # classic exam (no B)
    python scripts/guided_session.py --script data --person ana   # collect training data

With --script data, every button and combination (B included) is asked for in a
random order, and the script is saved in the file so each frame can be labelled.
--person is a short nickname to tell people apart in the dataset (never a full
name). Only hand landmarks are saved, never images.
"""

import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_live import (  # noqa: E402  (sibling script, same directory)
    MODEL_PATH,
    LiveCalibration,
    detect_hands,
)

from air_controller.capture import LatestFrameCamera, VideoFileSource  # noqa: E402
from air_controller.config import load_config  # noqa: E402
from air_controller.dashboard import WINDOW, draw_hands, render_game  # noqa: E402
from air_controller.guided import (  # noqa: E402
    build_data_script,
    build_script,
    script_arrays,
    to_mask,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "guided"
LEAD_IN_MS = 3000


class GuidedSession:
    def __init__(self, app: LiveCalibration, blocks: int, script_name: str = "classic",
                 seed: int = 0, person: str = "") -> None:
        self.app = app
        self.script_name = script_name
        self.seed = seed
        self.person = person
        self.script = (build_data_script(blocks, seed) if script_name == "data"
                       else build_script(blocks))
        self.rows: dict[str, list] = {key: [] for key in (
            "timestamp_ms", "step", "raw_mask", "held_mask", "left_visible", "right_visible",
            "dpad_left_progress", "dpad_right_progress", "posture_warning", "left", "right")}
        self.events: list[tuple[int, str, bool]] = []
        self.offsets_ms = np.cumsum([0] + [step.duration_ms for _, step in self.script])
        self.last_index = -1
        self.start_ms: int | None = None

    def run(self) -> bool:
        """True if the session reached the end; False if aborted with Q."""
        empty = np.full((21, 3), np.nan)
        while True:
            frame = self.app.source.read()
            if frame is None:
                return False
            now = frame.timestamp_ms
            if self.start_ms is None:
                self.start_ms = now + LEAD_IN_MS
            elapsed = now - self.start_ms
            index = self._step_index(elapsed)
            if index is not None and index >= len(self.script):
                return True

            image = frame.image
            left, right = detect_hands(self.app.landmarker, image, now)
            draw_hands(image, left, right)
            reading = self.app.classifier.classify(left, right, now)
            events = self.app.machine.update(reading, now)
            for event in events:
                self.events.append((event.timestamp_ms, event.button.value, event.pressed))
            self.app.count_presses(events)
            warnings = self.app.posture.update(left, right, now)

            if index is not None:
                self._record(now, index, reading, warnings, left, right, empty)
            canvas = self._draw(image, elapsed, index, reading, warnings)
            if self.app.show:
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False

    def _step_index(self, elapsed: int) -> int | None:
        """Index of the current step (len(script) when finished); None during the countdown."""
        if elapsed < 0:
            return None
        index = int(np.searchsorted(self.offsets_ms, elapsed, side="right")) - 1
        self.last_index = max(self.last_index, min(index, len(self.script) - 1))
        return index

    def _record(self, now, index, reading, warnings, left, right, empty) -> None:
        held = self.app.machine.held
        self.rows["timestamp_ms"].append(now)
        self.rows["step"].append(index)
        self.rows["raw_mask"].append(to_mask(reading.pressed))
        self.rows["held_mask"].append(to_mask(held))
        self.rows["left_visible"].append(reading.left_visible)
        self.rows["right_visible"].append(reading.right_visible)
        self.rows["dpad_left_progress"].append(reading.dpad_left_progress)
        self.rows["dpad_right_progress"].append(reading.dpad_right_progress)
        self.rows["posture_warning"].append(bool(warnings))
        self.rows["left"].append(empty if left is None else left)
        self.rows["right"].append(empty if right is None else right)

    def _draw(self, image, elapsed, index, reading, warnings):
        if index is None:
            banner = (f"Starting in {-elapsed / 1000:.0f}...", f"First: {self.script[0][1].text}")
            progress = None
        else:
            block, step = self.script[index]
            following = (f"Next: {self.script[index + 1][1].text}"
                         if index + 1 < len(self.script) else "Last step")
            banner = (step.text, following,
                      f"Block {block}/{self.script[-1][0]}   step {index + 1}/{len(self.script)}")
            progress = (elapsed - self.offsets_ms[index]) / step.duration_ms
        return render_game(image, self.app.game_view(
            reading, warnings, status="GUIDED", banner=banner, banner_progress=progress,
            hints=("Q  stop and save",)))

    def save(self, completed: bool) -> Path:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.person}" if self.person else ""
        path = OUT_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}.npz"
        rows = {key: np.array(values) for key, values in self.rows.items()}
        np.savez_compressed(
            path,
            completed=np.array(completed),
            blocks=np.array(max((b for b, _ in self.script), default=0)),
            step_starts_ms=np.array([self.start_ms + offset
                                     for offset in self.offsets_ms[:self.last_index + 1]]),
            event_times_ms=np.array([t for t, _, _ in self.events]),
            event_buttons=np.array([b for _, b, _ in self.events]),
            event_pressed=np.array([p for _, _, p in self.events]),
            press_confirm_ms=np.array(self.app.config.timing.press_confirm_ms),
            dpad_threshold=np.array(self.app.config.classifier.dpad_threshold),
            script_name=np.array(self.script_name),
            seed=np.array(self.seed),
            person=np.array(self.person),
            **script_arrays(self.script),
            **rows,
        )
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 guided session.")
    parser.add_argument("--video", type=Path, help="use a video instead of the camera (testing)")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--script", choices=("classic", "data"), default="classic",
                        help="classic = Phase 2 exam; data = every button incl. B, random order")
    parser.add_argument("--person", default="",
                        help="short nickname for the dataset, e.g. mateo or p01 (not a full name)")
    parser.add_argument("--seed", type=int, help="order of the data script (random if omitted)")
    args = parser.parse_args()
    if args.person and not re.fullmatch(r"[a-z0-9]{1,16}", args.person):
        print("--person must be a short nickname: lowercase letters and digits, e.g. mateo or p01.")
        return
    if args.script == "data" and not args.person:
        print("Data sessions need --person (a short nickname), so the dataset can tell people apart.")
        return
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    if args.script == "data":
        print(f"Collecting data for '{args.person}' (order seed {seed}). Only hand landmarks are "
              "saved, never images.")
    if not MODEL_PATH.exists():
        print(f"Model missing at {MODEL_PATH}. Download it as explained in the README (Quick start).")
        return
    config = load_config()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    session = None
    try:
        # Phase 1: calibrate until accepted (R repeats, Q quits).
        while not app.ready:
            if not app.step():
                print("Calibration interrupted; no session.")
                return
        print("\nCalibration accepted. Starting the guided session.")
        session = GuidedSession(app, args.blocks, args.script, seed, args.person)
        completed = session.run()
        app.machine.release_all(app.last_timestamp_ms)
        path = session.save(completed)
        status = "complete" if completed else "interrupted (saved anyway)"
        print(f"Session {status}: {path.relative_to(ROOT)}")
        print("Analyse it with: python scripts/evaluate_guided.py")
    finally:
        app.close()


if __name__ == "__main__":
    main()
