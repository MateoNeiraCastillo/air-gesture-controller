import numpy as np
import pytest

from air_controller.calibration import (
    CALIBRATION_SEQUENCE,
    CalibrationCollector,
    Pose,
    Reference,
    separation,
)
from air_controller.features import INDEX_MCP, MIDDLE_MCP, THUMB_TIP, WRIST

FRAMES = 20
RNG = np.random.default_rng(0)


def make_hand(tip_x: float, tip_y: float, noise: float = 0.5) -> np.ndarray:
    """Synthetic 100 px hand; only the wrist, finger bases and thumb tip matter."""
    hand = np.zeros((21, 3))
    hand[WRIST] = [0, 100, 0]
    hand[MIDDLE_MCP] = [0, 0, 0]
    hand[INDEX_MCP] = [10, 0, 0]
    hand[THUMB_TIP] = [tip_x, tip_y, 0]
    return hand + RNG.normal(0, noise, hand.shape)


# Clearly distinct synthetic poses (thumb tip, in pixels).
REST = (60, -40)
PRESSED = (60, 10)
DPAD_LEFT = (20, -40)
DPAD_RIGHT = (100, -40)


def feed(collector: CalibrationCollector, pose: Pose, left_tip, right_tip, frames: int = FRAMES) -> None:
    for _ in range(frames):
        if collector.current_pose is not pose:
            return
        collector.add(make_hand(*left_tip), make_hand(*right_tip))


def calibrate(pressed=PRESSED, dpad_left=DPAD_LEFT, dpad_right=DPAD_RIGHT,
              left_while_a=REST) -> CalibrationCollector:
    collector = CalibrationCollector(frames_per_pose=FRAMES)
    feed(collector, Pose.REST, REST, REST)
    feed(collector, Pose.A_PRESSED, left_while_a, pressed)
    feed(collector, Pose.DPAD_LEFT, dpad_left, REST)
    feed(collector, Pose.DPAD_RIGHT, dpad_right, REST)
    return collector


def test_follows_the_sequence_and_finishes() -> None:
    collector = CalibrationCollector(frames_per_pose=FRAMES)
    seen = []
    while not collector.done:
        seen.append(collector.current_pose)
        feed(collector, collector.current_pose, REST, REST)
    assert seen == list(CALIBRATION_SEQUENCE)
    assert collector.progress == 1.0


def test_good_calibration_is_accepted() -> None:
    result = calibrate().build(min_separation=3.0, dpad_threshold=0.5)
    assert result.ok, result.problems
    assert result.calibration.right_hand_size == pytest.approx(100, abs=2)


def test_weak_press_is_rejected_with_a_useful_message() -> None:
    result = calibrate(pressed=(60, -39)).build(min_separation=3.0, dpad_threshold=0.5)
    assert not result.ok
    assert any("Button A" in problem for problem in result.problems)


def test_weak_dpad_right_is_rejected() -> None:
    result = calibrate(dpad_right=(61, -40)).build(min_separation=3.0, dpad_threshold=0.5)
    assert not result.ok
    assert any("D-pad right" in problem for problem in result.problems)


def test_left_thumb_drifting_to_right_while_pressing_a_is_rejected() -> None:
    """The night-session case: when pressing A, the left thumb drifted to 'right'."""
    drifted = (REST[0] + 0.8 * (DPAD_RIGHT[0] - REST[0]), REST[1])
    result = calibrate(left_while_a=drifted).build(min_separation=3.0, dpad_threshold=0.5)
    assert not result.ok
    assert any("When pressing A" in problem and "right" in problem for problem in result.problems)


def test_frames_without_the_required_hand_do_not_count() -> None:
    collector = CalibrationCollector(frames_per_pose=FRAMES)
    assert collector.add(None, make_hand(*REST)) is False          # rest needs both
    assert collector.add(make_hand(*REST), make_hand(*REST)) is True
    assert collector.progress == pytest.approx(1 / FRAMES)


def test_nan_landmarks_do_not_count() -> None:
    collector = CalibrationCollector(frames_per_pose=FRAMES)
    broken = make_hand(*REST)
    broken[THUMB_TIP] = np.nan
    assert collector.add(broken, make_hand(*REST)) is False


def test_building_before_finishing_reports_a_problem() -> None:
    result = CalibrationCollector(frames_per_pose=FRAMES).build(min_separation=3.0, dpad_threshold=0.5)
    assert not result.ok


def test_reference_ignores_a_wild_outlier() -> None:
    samples = np.array([[1.0], [1.1], [0.9], [1.0], [1.05], [500.0]])
    reference = Reference.from_samples(samples)
    assert reference.center[0] == pytest.approx(1.0, abs=0.05)
    assert reference.spread[0] < 1.0


def test_separation_grows_with_distance() -> None:
    base = Reference(np.array([0.0]), np.array([1.0]))
    near = Reference(np.array([1.0]), np.array([1.0]))
    far = Reference(np.array([10.0]), np.array([1.0]))
    assert separation(base, near) < separation(base, far)
    assert separation(base, far) == pytest.approx(10.0)
