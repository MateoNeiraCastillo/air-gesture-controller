"""Live calibration and a view of what the classifier reads.

1. Waits until both hands are in view, then guides you through the 5 poses
   (rest, A pressed, B pressed, D-pad left, D-pad right). Each one starts with a
   countdown in the center of the screen.
2. If the calibration fails quality control, it tells you why
   (on screen and in the terminal). Press R to repeat.
3. If it passes, it shows SUCCESS and then the dashboard (air_controller/dashboard.py).
   Every confirmed press and release is printed in the terminal.

Keys: R = recalibrate, Q = quit.

On exit it saves the calibration in data/live/ so it can be analysed.

Usage:
    python scripts/calibrate_live.py
    python scripts/calibrate_live.py --video data/raw/something.mp4   # no camera
"""

import argparse
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from air_controller.buttons import Button
from air_controller.calibration import (
    CALIBRATION_SEQUENCE,
    REQUIRED_HANDS,
    CalibrationCollector,
    CalibrationResult,
    Pose,
    left_thumb_drift,
)
from air_controller.capture import FrameSource, LatestFrameCamera, VideoFileSource
from air_controller.classifier import Classifier, FrameReading
from air_controller.config import Config, load_config
from air_controller.dashboard import (
    GREEN,
    WINDOW,
    CalibrationView,
    GameView,
    draw_hands,
    open_window,
    render_calibration,
    render_game,
)
from air_controller.hands import assign_hands
from air_controller.posture import PostureMonitor
from air_controller.rest_tracker import AdaptiveClassifier, RestTracker
from air_controller.state_machine import ButtonStateMachine

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
LIVE_DIR = ROOT / "data" / "live"
PREPARE_SECONDS = 4.0
POSITION_SECONDS = 1.0   # both hands in view for this long before the first pose
SUCCESS_SECONDS = 2.5

POSITION_TITLE = "GET INTO POSITION"
POSITION_LINES = ("Raise both hands in front of the camera,",
                  "as if you were holding a game controller.",
                  "Left hand on the left of the screen, right hand on the right.",
                  "Calibration starts by itself when both hands are seen.")

INSTRUCTIONS = {
    Pose.REST: ("RELAX BOTH THUMBS",
                ("Hold the invisible controller with both hands.",
                 "Thumbs relaxed, pressing nothing. Then stay still.")),
    Pose.A_PRESSED: ("RIGHT THUMB: PRESS A",
                     ("Push your right thumb all the way down and hold it there.",
                      "Keep your LEFT thumb completely still.")),
    Pose.B_PRESSED: ("RIGHT THUMB: PRESS B",
                     ("Tilt your right thumb OUTWARDS, away from the other hand, and hold it.",
                      "Keep your LEFT thumb completely still.")),
    Pose.DPAD_LEFT: ("LEFT THUMB: PUSH LEFT",
                     ("Move your left thumb as far left as is comfortable,",
                      "and hold it there.")),
    Pose.DPAD_RIGHT: ("LEFT THUMB: PUSH RIGHT",
                      ("Move your left thumb as far right as is comfortable,",
                       "and hold it there. Make it a big, clear movement.")),
}


# --------------------------------------------------------------------------- #
# MediaPipe
# --------------------------------------------------------------------------- #

def build_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH),
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: vision.HandLandmarker, image: np.ndarray,
                 timestamp_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Landmarks (21, 3) in pixels for the left and right hand, or None."""
    rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(rgb, timestamp_ms)
    assigned = assign_hands(result.hand_landmarks)
    height, width = image.shape[:2]

    def to_pixels(hand) -> np.ndarray | None:
        if hand is None:
            return None
        return np.array([[p.x * width, p.y * height, p.z * width] for p in hand])

    return to_pixels(assigned.left), to_pixels(assigned.right)


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

class LiveCalibration:
    def __init__(self, source: FrameSource, config: Config, show: bool = True) -> None:
        self.source = source
        self.config = config
        self.show = show
        self.landmarker = build_landmarker()
        self.collector = CalibrationCollector(config.calibration.frames_per_pose)
        self.result: CalibrationResult | None = None
        self.classifier: Classifier | None = None
        self.pose_started_at: float | None = None
        self.current_pose: Pose | None = None
        self.positioned = False
        self.both_hands_since: float | None = None
        self.success_until = 0.0
        self.readings: list[FrameReading] = []
        self.machine = ButtonStateMachine(config.timing)
        self.posture: PostureMonitor | None = None
        self.presses: Counter[Button] = Counter()
        self.last_timestamp_ms = 0
        if show:
            open_window()

    @property
    def ready(self) -> bool:
        """True once the calibration is accepted and the SUCCESS screen has been shown."""
        return (self.result is not None and self.result.ok
                and time.perf_counter() >= self.success_until)

    def restart(self) -> None:
        self._log_events(self.machine.release_all(self.last_timestamp_ms))
        self.collector = CalibrationCollector(self.config.calibration.frames_per_pose)
        self.result = None
        self.classifier = None
        self.current_pose = None
        self.positioned = False
        self.both_hands_since = None
        self.presses.clear()
        print("\n--- Calibration restarted ---")

    def step(self) -> bool:
        """Processes one frame. False when it is time to quit."""
        frame = self.source.read()
        if frame is None:
            return False
        image = frame.image
        self.last_timestamp_ms = frame.timestamp_ms
        left, right = detect_hands(self.landmarker, image, frame.timestamp_ms)
        draw_hands(image, left, right)

        if self.result is None:
            canvas = self._calibrating(image, left, right)
        elif not self.result.ok:
            canvas = render_calibration(image, CalibrationView(
                "rejected", lines=tuple(self.result.problems), step=len(CALIBRATION_SEQUENCE),
                left_seen=left is not None, right_seen=right is not None))
        elif not self.ready:
            canvas = render_calibration(image, CalibrationView(
                "success", lines=("Calibration complete. Starting the controller...",),
                step=len(CALIBRATION_SEQUENCE),
                left_seen=left is not None, right_seen=right is not None))
        else:
            reading = self.classifier.classify(left, right, frame.timestamp_ms)
            self.readings.append(reading)
            events = self.machine.update(reading, frame.timestamp_ms)
            self._log_events(events)
            self.count_presses(events)
            warnings = self.posture.update(left, right, frame.timestamp_ms)
            canvas = render_game(image, self.game_view(reading, warnings))

        return self._show(canvas)

    def count_presses(self, events) -> None:
        for event in events:
            if event.pressed:
                self.presses[event.button] += 1

    def game_view(self, reading: FrameReading, warnings: list[str], **overrides) -> GameView:
        """What the dashboard shows. By default, what is confirmed counts as sent."""
        values = dict(reading=reading, held=self.machine.held, sent=self.machine.held,
                      keys=self.config.keys, threshold=self.config.classifier.dpad_threshold,
                      presses=self.presses, status="LIVE", status_color=GREEN,
                      warnings=tuple(warnings))
        return GameView(**{**values, **overrides})

    def _calibrating(self, image: np.ndarray, left, right) -> np.ndarray:
        seen = dict(left_seen=left is not None, right_seen=right is not None)
        now = time.perf_counter()
        if not self.positioned:
            both = left is not None and right is not None
            if not both:
                self.both_hands_since = None
            elif self.both_hands_since is None:
                self.both_hands_since = now
            if not both or now - self.both_hands_since < POSITION_SECONDS:
                return render_calibration(image, CalibrationView(
                    "position", POSITION_TITLE, POSITION_LINES, **seen))
            self.positioned = True

        pose = self.collector.current_pose
        if pose is not self.current_pose:
            self.current_pose = pose
            self.pose_started_at = now
            print(f"Pose: {pose.value}")
        title, lines = INSTRUCTIONS[pose]
        needed = dict(left_needed="left" in REQUIRED_HANDS[pose],
                      right_needed="right" in REQUIRED_HANDS[pose])
        number = CALIBRATION_SEQUENCE.index(pose) + 1
        waiting = PREPARE_SECONDS - (now - self.pose_started_at)
        if waiting > 0:
            return render_calibration(image, CalibrationView(
                "prepare", title, lines, number, countdown=math.ceil(waiting), **seen, **needed))

        self.collector.add(left, right)
        # The collector moves on by itself: this pose is complete if it no longer is the current one.
        progress = 1.0 if self.collector.current_pose is not pose else self.collector.progress
        if self.collector.done:
            self._finish_calibration()
        return render_calibration(image, CalibrationView(
            "capture", title, lines, number, progress=progress, **seen, **needed))

    def _finish_calibration(self) -> None:
        self.result = self.collector.build(self.config.calibration.min_separation,
                                           self.config.classifier.dpad_threshold)
        if self.result.ok:
            self.classifier = AdaptiveClassifier(Classifier(self.result.calibration, self.config.classifier),
                                                 RestTracker.from_config(self.config.classifier))
            self.posture = PostureMonitor(self.result.calibration, self.config.posture)
            self.machine = ButtonStateMachine(self.config.timing)
            self.presses.clear()
            self.success_until = time.perf_counter() + SUCCESS_SECONDS
            print("Calibration ACCEPTED.")
            for name, progress in left_thumb_drift(self.result.calibration).items():
                print(f"  Left thumb when pressing {name}: {progress[Button.LEFT]:.0%} towards left, "
                      f"{progress[Button.RIGHT]:.0%} towards right (limit "
                      f"{self.config.classifier.dpad_threshold / 2:.0%})")
        else:
            print("Calibration REJECTED:", *self.result.problems, sep="\n  ")
        self._save_calibration()

    def _save_calibration(self) -> None:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arrays = {f"{pose.name}_{hand}": values
                  for pose, by_hand in self.collector.samples().items()
                  for hand, values in by_hand.items()}
        path = LIVE_DIR / f"calibration_{stamp}.npz"
        np.savez(path, accepted=np.array(self.result.ok),
                 problems=np.array(self.result.problems, dtype=object), **arrays)
        print(f"  Saved to {path.relative_to(ROOT)}")

    def _show(self, image: np.ndarray) -> bool:
        if not self.show:
            return True
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("r"):
            self.restart()
        return True

    def _log_events(self, events) -> None:
        for event in events:
            action = "PRESS  " if event.pressed else "release"
            print(f"  {event.timestamp_ms / 1000:8.2f} s  {action} {event.button.value}")

    def close(self) -> None:
        self._log_events(self.machine.release_all(self.last_timestamp_ms))
        self.source.close()
        self.landmarker.close()
        if self.show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live calibration.")
    parser.add_argument("--video", type=Path, help="use a video instead of the camera")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        print(f"Model missing at {MODEL_PATH}. Download it as explained in the README (Quick start).")
        return
    config = load_config()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    try:
        while app.step():
            pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
