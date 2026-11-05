import numpy as np
import pytest

from air_controller.calibration import CalibrationCollector, Pose
from air_controller.config import PostureConfig
from air_controller.features import INDEX_MCP, MIDDLE_MCP, THUMB_TIP, WRIST
from air_controller.posture import PostureMonitor

CONFIG = PostureConfig(hand_size_ratio_min=0.7, hand_size_ratio_max=1.4, warning_after_ms=500)
RNG = np.random.default_rng(2)


def make_hand(tip, size: float = 100.0) -> np.ndarray:
    hand = np.zeros((21, 3))
    hand[WRIST] = [0, size, 0]
    hand[MIDDLE_MCP] = [0, 0, 0]
    hand[INDEX_MCP] = [0.1 * size, 0, 0]
    hand[THUMB_TIP] = [tip[0] * size / 100, tip[1] * size / 100, 0]
    return hand + RNG.normal(0, 0.5, hand.shape)


@pytest.fixture(scope="module")
def monitor_factory():
    collector = CalibrationCollector(frames_per_pose=20)
    plan = [(Pose.REST, (60, -40), (60, -40)), (Pose.A_PRESSED, (60, -40), (60, 10)),
            (Pose.B_PRESSED, (60, -40), (100, -40)),
            (Pose.DPAD_LEFT, (20, -40), (60, -40)), (Pose.DPAD_RIGHT, (100, -40), (60, -40))]
    for pose, left_tip, right_tip in plan:
        while collector.current_pose is pose:
            collector.add(make_hand(left_tip), make_hand(right_tip))
    calibration = collector.build(min_separation=3.0, dpad_threshold=0.5).calibration
    return lambda: PostureMonitor(calibration, CONFIG)


def test_same_posture_gives_no_warning(monitor_factory) -> None:
    monitor = monitor_factory()
    for t in range(0, 2000, 40):
        assert monitor.update(make_hand((60, -40)), make_hand((60, -40)), t) == []


def test_rotated_hand_warns_only_after_delay(monitor_factory) -> None:
    monitor = monitor_factory()
    rotated = make_hand((60, -40), size=30)          # the daytime rhythm case
    assert monitor.update(make_hand((60, -40)), rotated, 0) == []
    assert monitor.update(make_hand((60, -40)), rotated, 400) == []
    warnings = monitor.update(make_hand((60, -40)), rotated, 520)
    assert len(warnings) == 1 and "Right" in warnings[0] and "rotated" in warnings[0]


def test_brief_glitch_does_not_warn(monitor_factory) -> None:
    monitor = monitor_factory()
    monitor.update(make_hand((60, -40), size=30), make_hand((60, -40)), 0)
    assert monitor.update(make_hand((60, -40)), make_hand((60, -40)), 40) == []
    assert monitor.update(make_hand((60, -40)), make_hand((60, -40)), 1000) == []


def test_hand_too_close_warns(monitor_factory) -> None:
    monitor = monitor_factory()
    big = make_hand((60, -40), size=200)
    monitor.update(big, make_hand((60, -40)), 0)
    warnings = monitor.update(big, make_hand((60, -40)), 600)
    assert len(warnings) == 1 and "Left" in warnings[0] and "close" in warnings[0]


def test_missing_hand_is_not_a_posture_warning(monitor_factory) -> None:
    monitor = monitor_factory()
    monitor.update(None, None, 0)
    assert monitor.update(None, None, 1000) == []
