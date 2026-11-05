import math

import numpy as np
import pytest

from air_controller.buttons import Button
from air_controller.calibration import CalibrationCollector, Pose
from air_controller.classifier import Classifier
from air_controller.config import ClassifierConfig
from air_controller.features import INDEX_MCP, MIDDLE_MCP, THUMB_TIP, WRIST

FRAMES = 20
RNG = np.random.default_rng(1)

REST = (60, -40)
PRESSED = (60, 10)
DPAD_LEFT = (20, -40)
DPAD_RIGHT = (100, -40)


def make_hand(tip_x: float, tip_y: float, noise: float = 0.5) -> np.ndarray:
    hand = np.zeros((21, 3))
    hand[WRIST] = [0, 100, 0]
    hand[MIDDLE_MCP] = [0, 0, 0]
    hand[INDEX_MCP] = [10, 0, 0]
    hand[THUMB_TIP] = [tip_x, tip_y, 0]
    return hand + RNG.normal(0, noise, hand.shape)


@pytest.fixture(scope="module")
def classifier() -> Classifier:
    collector = CalibrationCollector(frames_per_pose=FRAMES)
    plan = [(Pose.REST, REST, REST), (Pose.A_PRESSED, REST, PRESSED),
            (Pose.DPAD_LEFT, DPAD_LEFT, REST), (Pose.DPAD_RIGHT, DPAD_RIGHT, REST)]
    for pose, left_tip, right_tip in plan:
        while collector.current_pose is pose:
            collector.add(make_hand(*left_tip), make_hand(*right_tip))
    result = collector.build(min_separation=3.0, dpad_threshold=0.5)
    assert result.ok, result.problems
    return Classifier(result.calibration, ClassifierConfig(dpad_threshold=0.5))


def between(a: tuple[float, float], b: tuple[float, float], fraction: float) -> tuple[float, float]:
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))


def test_rest_presses_nothing(classifier: Classifier) -> None:
    reading = classifier.classify(make_hand(*REST), make_hand(*REST))
    assert reading.pressed == frozenset()
    assert reading.left_visible and reading.right_visible


def test_a_pressed(classifier: Classifier) -> None:
    assert classifier.classify(make_hand(*REST), make_hand(*PRESSED)).pressed == {Button.A}


@pytest.mark.parametrize("target,button", [(DPAD_LEFT, Button.LEFT), (DPAD_RIGHT, Button.RIGHT)])
def test_full_dpad_push(classifier: Classifier, target, button: Button) -> None:
    assert classifier.classify(make_hand(*target), make_hand(*REST)).pressed == {button}


def test_weaker_push_than_calibration_still_counts(classifier: Classifier) -> None:
    """The daytime combo case: pushing at 70 % of the calibrated travel is still 'right'."""
    reading = classifier.classify(make_hand(*between(REST, DPAD_RIGHT, 0.7)), make_hand(*REST))
    assert reading.pressed == {Button.RIGHT}
    assert reading.dpad_right_progress == pytest.approx(0.7, abs=0.1)


def test_small_drift_is_ignored(classifier: Classifier) -> None:
    reading = classifier.classify(make_hand(*between(REST, DPAD_RIGHT, 0.3)), make_hand(*REST))
    assert Button.RIGHT not in reading.pressed


def test_dpad_and_a_together(classifier: Classifier) -> None:
    reading = classifier.classify(make_hand(*DPAD_RIGHT), make_hand(*PRESSED))
    assert reading.pressed == {Button.RIGHT, Button.A}


def test_missing_hand_presses_nothing_on_that_side(classifier: Classifier) -> None:
    reading = classifier.classify(None, make_hand(*PRESSED))
    assert reading.pressed == {Button.A}
    assert not reading.left_visible
    assert math.isnan(reading.dpad_right_progress)


def test_nan_landmarks_count_as_missing(classifier: Classifier) -> None:
    broken = make_hand(*PRESSED)
    broken[THUMB_TIP] = np.nan
    reading = classifier.classify(make_hand(*REST), broken)
    assert not reading.right_visible
    assert Button.A not in reading.pressed
