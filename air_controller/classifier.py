"""Layer 3: which buttons are pressed in THIS frame, according to the calibration.

Pure, memoryless logic: one frame in, one reading out. Smoothing over time
(not releasing a button because of one odd frame) is the job of the state
machine (layer 4).

- Button A: is the right thumb tip closer to the calibrated "pressed" or
  "rest"? Distances measured in noise widths.
- D-pad, in two steps:
    1. Direction: the nearest calibrated pose (left or right).
    2. Intensity: how far the thumb has moved from rest towards it.
       Counts as pressed from `dpad_threshold` of the travel onwards.
  This way, pushing more lightly than during calibration still works (in the
  spike, the daytime `combo` was pushed less than `dpad_right`). The direction
  is NOT chosen by "largest progress": in the night session left and right
  pointed the same way (cosine +0.95) and right was a short travel; a "left"
  point progressed 3 times further along right's short axis.
"""

from dataclasses import dataclass

import numpy as np

from air_controller.buttons import Button
from air_controller.calibration import BUTTON_COLUMNS, DPAD_COLUMNS, Calibration, Reference
from air_controller.config import ClassifierConfig
from air_controller.features import thumb_position_features


@dataclass(frozen=True)
class FrameReading:
    """Result of classifying a frame."""

    pressed: frozenset[Button]
    left_visible: bool
    right_visible: bool
    dpad_left_progress: float    # 0 = rest, 1 = as in calibration; NaN if not visible
    dpad_right_progress: float
    dpad_direction: Button | None = None   # nearest calibrated direction; None if not visible


@dataclass(frozen=True)
class DpadAxis:
    """One D-pad direction, precomputed from the calibration."""

    unit: np.ndarray        # direction rest → target, in scaled space
    length: float           # distance rest → target, in the same space

    def progress(self, offset: np.ndarray) -> float:
        """Fraction of the calibrated travel covered by `offset` in this direction."""
        return float(offset @ self.unit) / self.length


class Classifier:
    def __init__(self, calibration: Calibration, config: ClassifierConfig) -> None:
        self._calibration = calibration
        self._threshold = config.dpad_threshold
        self._button_scale = _pooled_spread(calibration.right_rest, calibration.right_pressed)

    def classify(self, left: np.ndarray | None, right: np.ndarray | None) -> FrameReading:
        """`left`/`right`: landmarks (21, 3) in pixels, or None if the hand is not visible."""
        pressed: set[Button] = set()
        left_features = _features(left)
        right_features = _features(right)

        if right_features is not None and self._a_pressed(right_features):
            pressed.add(Button.A)

        progress = {Button.LEFT: float("nan"), Button.RIGHT: float("nan")}
        nearest = None
        if left_features is not None:
            point = left_features[list(DPAD_COLUMNS)]
            progress = self._calibration.dpad.progress(point)
            nearest = self._calibration.dpad.nearest(point)
            if progress[nearest] >= self._threshold:
                pressed.add(nearest)

        return FrameReading(
            pressed=frozenset(pressed),
            left_visible=left_features is not None,
            right_visible=right_features is not None,
            dpad_left_progress=progress[Button.LEFT],
            dpad_right_progress=progress[Button.RIGHT],
            dpad_direction=nearest,
        )

    def _a_pressed(self, features: np.ndarray) -> bool:
        point = features[list(BUTTON_COLUMNS)]
        to_rest = np.linalg.norm((point - self._calibration.right_rest.center) / self._button_scale)
        to_pressed = np.linalg.norm((point - self._calibration.right_pressed.center) / self._button_scale)
        return bool(to_pressed < to_rest)


def _features(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None or np.isnan(landmarks).any():
        return None
    features = thumb_position_features(landmarks)
    return None if np.isnan(features).any() else features


def _pooled_spread(*references: Reference) -> np.ndarray:
    """Common scale: root mean square of the poses' spreads."""
    return np.sqrt(np.mean([reference.spread ** 2 for reference in references], axis=0))
