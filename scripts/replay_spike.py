"""Runs the spike clips through calibration + classifier, frame by frame.

For each session (night, day, ...):
  1. Calibrates with the first valid frames of rest, A held,
     D-pad left and D-pad right (as the game would at startup).
  2. Classifies ALL the clips of that session, skipping the frames used
     for calibration.
  3. Compares against what was asked for in each clip.

This is the classifier test without a camera. The percentages are per frame,
raw; presses are also counted AFTER the state machine, which is what would
reach the game. At the end, the posture monitor warnings.

Usage:
    python scripts/replay_spike.py
    python scripts/replay_spike.py --ignore-quality   # evaluate even if calibration is rejected
"""

import argparse

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air_controller.buttons import Button
from air_controller.calibration import Calibration, CalibrationCollector, Pose
from air_controller.classifier import Classifier, FrameReading
from air_controller.config import load_config
from air_controller.posture import PostureMonitor
from air_controller.state_machine import ButtonStateMachine

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = ROOT / "data" / "landmarks"
SETTLE_FRAMES = 30   # first second of each clip: still positioning the hands
FRAME_MS = 1000 / 30
TARGET = 95.0

CALIBRATION_CLIPS = {
    Pose.REST: "rest",
    Pose.A_PRESSED: "a_held",
    Pose.DPAD_LEFT: "dpad_left",
    Pose.DPAD_RIGHT: "dpad_right",
}


@dataclass(frozen=True)
class Clip:
    left: np.ndarray    # (frames, 21, 3), NaN where the hand wasn't seen
    right: np.ndarray

    def hands(self, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        left, right = self.left[index], self.right[index]
        return (None if np.isnan(left).any() else left,
                None if np.isnan(right).any() else right)

    def __len__(self) -> int:
        return len(self.left)


def load_sessions() -> dict[str, dict[str, Clip]]:
    """session → label → Clip (if there are several takes, the last one)."""
    sessions: dict[str, dict[str, Clip]] = defaultdict(dict)
    for path in sorted(LANDMARKS_DIR.glob("*.npz")):
        label, session, _index = path.stem.rsplit("_", 2)
        with np.load(path) as data:
            sessions[session][label] = Clip(data["left"][SETTLE_FRAMES:],
                                            data["right"][SETTLE_FRAMES:])
    return dict(sessions)


def calibrate(clips: dict[str, Clip], frames_per_pose: int, min_separation: float,
              ignore_quality: bool = False
              ) -> tuple[Classifier | None, Calibration | None, dict[str, int]]:
    """Returns the classifier and, per clip, how many frames the calibration consumed."""
    config = load_config()
    collector = CalibrationCollector(frames_per_pose)
    used: dict[str, int] = {}
    for pose, label in CALIBRATION_CLIPS.items():
        clip = clips[label]
        index = 0
        while collector.current_pose is pose and index < len(clip):
            collector.add(*clip.hands(index))
            index += 1
        used[label] = index
    result = collector.build(min_separation, config.classifier.dpad_threshold)
    if not result.ok:
        print("  Calibration rejected:", *result.problems, sep="\n    ")
        if not ignore_quality:
            return None, None, used
        print("  (--ignore-quality: evaluating anyway)")
        result = collector.build(min_separation=0.0, dpad_threshold=float("inf"))
    return Classifier(result.calibration, config.classifier), result.calibration, used


def readings(classifier: Classifier, clip: Clip, start: int) -> list[FrameReading]:
    return [classifier.classify(*clip.hands(index)) for index in range(start, len(clip))]


def share(values: list[FrameReading], condition) -> float:
    """% of frames meeting `condition`, among the frames with the relevant hand visible."""
    return 100 * float(np.mean([condition(r) for r in values])) if values else float("nan")


def presses(values: list[FrameReading], button: Button) -> int:
    on = np.array([button in r.pressed for r in values])
    return int(np.sum(on[1:] & ~on[:-1]))


def confirmed(values: list[FrameReading]) -> tuple[dict[Button, int], dict[Button, float]]:
    """Runs the readings through the state machine.

    Returns, per button, how many presses would be sent to the game and what %
    of the time it would be held.
    """
    machine = ButtonStateMachine(load_config().timing)
    counts = {button: 0 for button in Button}
    held_frames = {button: 0 for button in Button}
    for index, reading in enumerate(values):
        for event in machine.update(reading, int(index * FRAME_MS)):
            if event.pressed:
                counts[event.button] += 1
        for button in machine.held:
            held_frames[button] += 1
    total = max(len(values), 1)
    return counts, {button: 100 * frames / total for button, frames in held_frames.items()}


def posture_warning_share(calibration: Calibration, clip: Clip, start: int) -> float:
    monitor = PostureMonitor(calibration, load_config().posture)
    warned = [bool(monitor.update(*clip.hands(index), int(index * FRAME_MS)))
              for index in range(start, len(clip))]
    return 100 * float(np.mean(warned)) if warned else float("nan")


def report_session(session: str, clips: dict[str, Clip], ignore_quality: bool) -> list[float]:
    config = load_config()
    print(f"\n== Session '{session}' ==")
    if any(label not in clips for label in CALIBRATION_CLIPS.values()):
        print("  Calibration clips are missing.")
        return []
    classifier, calibration, used = calibrate(clips, config.calibration.frames_per_pose,
                                              config.calibration.min_separation, ignore_quality)
    if classifier is None:
        return []

    def run(label: str) -> list[FrameReading]:
        return readings(classifier, clips[label], used.get(label, 0))

    scores: list[float] = []

    def line(text: str, value: float, expected: bool = True) -> None:
        mark = "OK " if value >= TARGET else "NO "
        print(f"  {mark if expected else '   '}{text:<52}{value:6.1f} %")
        if expected:
            scores.append(value)

    right_seen = lambda r: r.right_visible
    left_seen = lambda r: r.left_visible

    rest = run("rest")
    line("rest: A not pressed", share([r for r in rest if right_seen(r)], lambda r: Button.A not in r.pressed))
    line("rest: D-pad not pressed", share([r for r in rest if left_seen(r)],
                                          lambda r: not ({Button.LEFT, Button.RIGHT} & r.pressed)))
    held = run("a_held")
    line("A held: A pressed", share([r for r in held if right_seen(r)], lambda r: Button.A in r.pressed))
    for label, button in (("dpad_left", Button.LEFT), ("dpad_right", Button.RIGHT)):
        values = [r for r in run(label) if left_seen(r)]
        line(f"{label}: only {button.value} on the D-pad", share(
            values, lambda r, b=button: b in r.pressed and len({Button.LEFT, Button.RIGHT} & r.pressed) == 1))

    for label in ("a_held", "a_rhythm"):
        if label in clips:
            values = [r for r in run(label) if left_seen(r)]
            line(f"{label}: D-pad not pressed (only A is used)", share(
                values, lambda r: not ({Button.LEFT, Button.RIGHT} & r.pressed)))

    if "combo" in clips:
        combo = run("combo")
        line("combo: D-pad on 'right'", share([r for r in combo if left_seen(r)],
                                              lambda r: Button.RIGHT in r.pressed))

    print("\n  After the state machine (what would reach the game):")
    for label in ("a_rhythm", "combo", "a_held"):
        if label not in clips:
            continue
        values = run(label)
        counts, held_pct = confirmed(values)
        seconds = len(values) / 30
        print(f"     {label:<18} A: {presses(values, Button.A):3d} raw → {counts[Button.A]:3d} "
              f"confirmed in {seconds:.0f} s | D-pad held: left "
              f"{held_pct[Button.LEFT]:5.1f} %, right {held_pct[Button.RIGHT]:5.1f} %")

    print("\n  Posture monitor (% of time with a warning):")
    for label in sorted(clips):
        print(f"     {label:<18} {posture_warning_share(calibration, clips[label], used.get(label, 0)):5.1f} %")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ignore-quality", action="store_true",
                        help="evaluate even if the calibration fails quality control")
    args = parser.parse_args()
    sessions = load_sessions()
    if not sessions:
        print(f"No landmarks in {LANDMARKS_DIR}.")
        return
    all_scores = [score for session, clips in sorted(sessions.items())
                  for score in report_session(session, clips, args.ignore_quality)]
    if all_scores:
        print(f"\nWorst case: {min(all_scores):.1f} %  (target >= {TARGET:.0f} %)")
        print("'raw' = frame by frame; 'confirmed' = after the state machine.")


if __name__ == "__main__":
    main()
