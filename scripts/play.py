"""The complete controller: play with the invisible controller.

Flow:
  1. Calibration, guided in the center of the screen (R repeats if rejected).
     SUCCESS is shown when it is accepted.
  2. Game: camera → hands → classifier → state machine → keys.
  3. On exit it prints the per-stage latency (p50 / p95).

Keys (with the camera window focused):
  P = pause / resume (releases all keys)
  R = recalibrate    (releases all keys)
  Q = quit
Emergency brake: move your hands away from the camera. After `missing_hand_ms`
(200 ms) all keys are released.

IMPORTANT: keys go to whichever window has focus. After calibrating, click
on the game window. On macOS, the app you run from (Terminal or VS Code)
needs the Accessibility permission.

Usage:
    python scripts/play.py --dry-run     # doesn't send keys: prints them
    python scripts/play.py               # plays for real
    python scripts/play.py --game celeste  # keys from the [games.celeste] profile
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_live import LiveCalibration, detect_hands  # noqa: E402  (sibling script, same directory)

from air_controller.buttons import Button  # noqa: E402
from air_controller.capture import Frame, LatestFrameCamera, VideoFileSource  # noqa: E402
from air_controller.config import keys_for_game, load_config  # noqa: E402
from air_controller.dashboard import GREEN, WINDOW, YELLOW, draw_hands, render_game  # noqa: E402
from air_controller.latency import LatencyTracker  # noqa: E402
from air_controller.output import KeyOutput, PynputKeySender, RecordingKeySender  # noqa: E402


class Player:
    def __init__(self, app: LiveCalibration, output: KeyOutput, keys: dict[Button, str],
                 show: bool = True, dry_run: bool = False) -> None:
        self.app = app
        self.output = output
        self.keys = keys
        self.show = show
        self.dry_run = dry_run
        self.paused = False
        self.latency = LatencyTracker()

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Calibrates and plays until Q (or until the video ends)."""
        while True:
            if not self._calibrate():
                return
            action = self._play()
            if action == "quit":
                return
            self.app.restart()                 # "recalibrate"

    def _calibrate(self) -> bool:
        while not self.app.ready:
            if not self.app.step():
                return False
        print("\nCalibration accepted. Let's play! Click on the game window.")
        return True

    def _play(self) -> str:
        """Game loop. Returns 'quit' or 'recalibrate'."""
        while True:
            frame = self.app.source.read()
            if frame is None:
                return "quit"
            self.app.last_timestamp_ms = frame.timestamp_ms
            image = self._process(frame)
            action = self._handle_keys(image)
            if action is not None:
                self._release_everything()
                return action

    def _process(self, frame: Frame):
        start = time.perf_counter()
        self.latency.add_frame(start)
        self.latency.add_stage("frame wait", (start - frame.captured_s) * 1000)

        left, right = detect_hands(self.app.landmarker, frame.image, frame.timestamp_ms)
        detected = time.perf_counter()
        self.latency.add_stage("detection (MediaPipe)", (detected - start) * 1000)

        reading = self.app.classifier.classify(left, right)
        events = self.app.machine.update(reading, frame.timestamp_ms)
        warnings = self.app.posture.update(left, right, frame.timestamp_ms)
        decided = time.perf_counter()
        self.latency.add_stage("classify + states", (decided - detected) * 1000)

        if not self.paused and events:
            self.output.apply(events)
            self.app.count_presses(events)
            sent = time.perf_counter()
            self.latency.add_stage("key sending", (sent - decided) * 1000)
            self.latency.add_end_to_end((sent - frame.captured_s) * 1000)

        image = frame.image
        draw_hands(image, left, right)
        return render_game(image, self._view(reading, warnings))

    def _view(self, reading, warnings):
        down = self.output.pressed_keys
        info = [f"{self.latency.fps():.0f} fps"]
        info.append("DRY RUN: no keys are sent, they are printed in the terminal." if self.dry_run
                    else "Click your game window: keys go to the window in focus.")
        info.append("To use P, R or Q, click this window first.")
        return self.app.game_view(
            reading, warnings,
            keys=self.keys,
            sent=frozenset(button for button, key in self.keys.items() if key in down),
            status="PAUSED" if self.paused else "PLAYING",
            status_color=YELLOW if self.paused else GREEN,
            info=tuple(info),
            hints=("P  pause", "R  recalibrate", "Q  quit"),
        )

    def _handle_keys(self, image) -> str | None:
        if not self.show:
            return None
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("r"):
            return "recalibrate"
        if key == ord("p"):
            self.paused = not self.paused
            if self.paused:
                self._release_everything()
            print("Paused" if self.paused else "Resumed")
        return None

    def _release_everything(self) -> None:
        """Releases in the state machine AND on the keyboard: nothing stays pressed."""
        self.app.machine.release_all(self.app.last_timestamp_ms)
        self.output.release_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Air Gesture Controller.")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't send keys: print them in the terminal")
    parser.add_argument("--video", type=Path, help="use a video instead of the camera (testing)")
    parser.add_argument("--game", help="key profile from config.toml ([games.<name>]), e.g. celeste")
    args = parser.parse_args()

    config = load_config()
    keys = keys_for_game(config, args.game)
    print("Keys: " + ", ".join(f"{button.value} → {key}" for button, key in keys.items()))
    sender = RecordingKeySender(echo=True) if args.dry_run else PynputKeySender()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    player = None
    try:
        with KeyOutput(sender, keys) as output:
            player = Player(app, output, keys, dry_run=args.dry_run)
            player.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        app.close()
        if player is not None:
            print(player.latency.summary())


if __name__ == "__main__":
    main()
