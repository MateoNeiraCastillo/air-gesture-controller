"""The complete controller: play with the invisible controller.

Flow:
  1. Calibration, guided in the center of the screen (R repeats if rejected).
     SUCCESS is shown when it is accepted.
  2. Game: camera → hands → classifier → state machine → keys, in their own
     thread (air_controller/runtime.py), so drawing the dashboard never delays
     a key.
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
    python scripts/play.py               # plays for real (default keys: Subway Surfers)
    python scripts/play.py --game celeste  # keys from a [games.<name>] profile in config.toml
    python scripts/analyze_play.py         # what went wrong in the last game
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_live import MODEL_PATH, LiveCalibration, detect_hands  # noqa: E402  (sibling script, same directory)

from air_controller.buttons import Button  # noqa: E402
from air_controller.capture import LatestFrameCamera, VideoFileSource  # noqa: E402
from air_controller.config import config_for_game, keys_for_game, load_config  # noqa: E402
from air_controller.dashboard import GREEN, WINDOW, YELLOW, draw_hands, render_game  # noqa: E402
from air_controller.latency import LatencyTracker  # noqa: E402
from air_controller.output import KeyOutput, PynputKeySender, RecordingKeySender  # noqa: E402
from air_controller.play_log import PlayLog  # noqa: E402
from air_controller.runtime import ControllerLoop, Snapshot  # noqa: E402

PLAY_DIR = Path(__file__).resolve().parent.parent / "data" / "play"


class Player:
    """Calibration in this thread; the game in two: keys in a `ControllerLoop` thread,
    the dashboard here (macOS only lets the main thread own OpenCV windows)."""

    def __init__(self, app: LiveCalibration, output: KeyOutput, keys: dict[Button, str],
                 show: bool = True, dry_run: bool = False, record: bool = False,
                 game: str | None = None) -> None:
        self.app = app
        self.output = output
        self.keys = keys
        self.show = show
        self.dry_run = dry_run
        self.latency = LatencyTracker()          # the key path, measured in the loop thread
        self.drawing = LatencyTracker()          # the dashboard, measured here
        self.record = record
        self.game = game
        self.saved: list[Path] = []              # recorded games (one per calibration)

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

    def summary(self) -> str:
        lines = [self.latency.summary(),
                 f"Dashboard: {self.drawing.fps():.1f} fps, drawing p50 "
                 f"{_median(self.drawing.stages.get('draw', []))} ms (in parallel, doesn't delay keys)"]
        for path in self.saved:
            lines.append(f"Game recorded in {path.relative_to(PLAY_DIR.parent.parent)}: "
                         "see what went wrong with  python scripts/analyze_play.py")
        return "\n".join(lines)

    def _calibrate(self) -> bool:
        while not self.app.ready:
            if not self.app.step():
                return False
        print("\nCalibration accepted. Let's play! Click on the game window.")
        return True

    def _play(self) -> str:
        """Game. Returns 'quit' or 'recalibrate'. Keys are always released on the way out."""
        loop = ControllerLoop(
            source=self.app.source,
            detect=lambda image, timestamp_ms: detect_hands(self.app.landmarker, image, timestamp_ms),
            classifier=self.app.classifier, machine=self.app.machine, posture=self.app.posture,
            output=self.output, latency=self.latency, log=PlayLog() if self.record else None)
        loop.start()
        action = "quit"
        try:
            action = self._show_until_done(loop)
        finally:
            loop.stop()                        # joins the thread, releases every key
            self._save(loop.log)
        return action

    def _save(self, log: PlayLog | None) -> None:
        if log is None or len(log) < 2:
            return
        path = PLAY_DIR / f"play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        config = self.app.config
        log.save(path, game=self.game or "", dry_run=self.dry_run,
                 dpad_threshold=config.classifier.dpad_threshold,
                 press_confirm_ms=config.timing.press_confirm_ms,
                 release_confirm_ms=config.timing.release_confirm_ms)
        self.saved.append(path)

    def _show_until_done(self, loop: ControllerLoop) -> str:
        paused = False
        while not loop.finished:
            snapshot = loop.wait_snapshot(timeout=0.05)
            if not self.show:
                continue
            if snapshot is not None and snapshot.sequence:
                self._draw(snapshot)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return "quit"
            if key == ord("r"):
                return "recalibrate"
            if key == ord("p"):
                paused = not paused
                if paused:
                    loop.pause()
                else:
                    loop.resume()
                print("Paused" if paused else "Resumed")
        return "quit"

    def _draw(self, snapshot: Snapshot) -> None:
        start = time.perf_counter()
        self.drawing.add_frame(start)
        image = snapshot.frame.image           # the loop never touches it again
        draw_hands(image, snapshot.left, snapshot.right)
        cv2.imshow(WINDOW, render_game(image, self._view(snapshot)))
        self.drawing.add_stage("draw", (time.perf_counter() - start) * 1000)

    def _view(self, snapshot: Snapshot):
        info = [f"{snapshot.fps:.0f} fps"]
        info.append("DRY RUN: no keys are sent, they are printed in the terminal." if self.dry_run
                    else "Click your game window: keys go to the window in focus.")
        info.append("To use P, R or Q, click this window first.")
        return self.app.game_view(
            snapshot.reading, list(snapshot.warnings),
            held=snapshot.held,
            presses=snapshot.presses,
            keys=self.keys,
            sent=frozenset(button for button, key in self.keys.items() if key in snapshot.pressed_keys),
            status="PAUSED" if snapshot.paused else "PLAYING",
            status_color=YELLOW if snapshot.paused else GREEN,
            info=tuple(info),
            hints=("P  pause", "R  recalibrate", "Q  quit"),
        )


def _median(values: list[float]) -> str:
    return f"{sorted(values)[len(values) // 2]:.1f}" if values else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="Air Gesture Controller.")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't send keys: print them in the terminal")
    parser.add_argument("--video", type=Path, help="use a video instead of the camera (testing)")
    parser.add_argument("--game", help="key profile from config.toml ([games.<name>]), e.g. celeste")
    parser.add_argument("--no-record", action="store_true",
                        help="don't save the game in data/play/ (it's saved by default for analyze_play.py)")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        print(f"Model missing at {MODEL_PATH}. Download it as explained in the README (Quick start).")
        return

    # OpenCV's drawing threads competed with MediaPipe for the CPU: on 05-11, drawing
    # in parallel raised MediaPipe from 42 to 56 ms per frame. Drawing on one thread
    # is a little slower, but drawing doesn't delay keys any more and MediaPipe does.
    cv2.setNumThreads(1)
    config = config_for_game(load_config(), args.game)
    keys = keys_for_game(config, args.game)
    print("Keys: " + ", ".join(f"{button.value} → {key}" for button, key in keys.items()))
    sender = RecordingKeySender(echo=True) if args.dry_run else PynputKeySender()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    player = None
    try:
        with KeyOutput(sender, keys) as output:
            player = Player(app, output, keys, dry_run=args.dry_run, record=not args.no_record,
                            game=args.game)
            player.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        app.close()
        if player is not None:
            print(player.summary())


if __name__ == "__main__":
    main()
