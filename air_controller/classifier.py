"""Layer 3: which buttons are pressed in THIS frame, according to the calibration.

Pure, memoryless logic: one frame in, one reading out. Smoothing over time
(not releasing a button because of one odd frame) is the job of the state
machine (layer 4).

- Buttons A and B (right thumb): each one on its own axis. A: is the tip's height
  closer to "A pressed" (thumb down) than to rest? B: is its sideways position
  closer to "B pressed" (thumb outwards) than to rest? Distances measured in
  noise widths. With a single thumb, A and B exclude each other: if both look
  pressed, the clearer one wins (in Celeste you jump and then dash).
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
from air_controller.calibration import A_COLUMNS, B_COLUMNS, DPAD_COLUMNS, Calibration, Reference
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


@dataclass(frozen=True)
class ThumbButton:
    """A right-thumb button, judged only on its own axis (A: down, B: outwards)."""

    button: Button
    columns: tuple[int, ...]
    rest: Reference
    pressed: Reference

    def margin(self, features: np.ndarray) -> float:
        """> 0 when the thumb is closer to "pressed" than to rest, in noise widths."""
        point = features[list(self.columns)]
        scale = _pooled_spread(self.rest.select(self.columns), self.pressed.select(self.columns))
        to_rest = np.linalg.norm((point - self.rest.center[list(self.columns)]) / scale)
        to_pressed = np.linalg.norm((point - self.pressed.center[list(self.columns)]) / scale)
        return float(to_rest - to_pressed)


class Classifier:
    def __init__(self, calibration: Calibration, config: ClassifierConfig) -> None:
        self._calibration = calibration
        self._threshold = config.dpad_threshold
        self._thumb_buttons = [ThumbButton(Button.A, A_COLUMNS, calibration.right_rest, calibration.right_a)]
        if calibration.right_b is not None:
            self._thumb_buttons.append(
                ThumbButton(Button.B, B_COLUMNS, calibration.right_rest, calibration.right_b))

    def classify(self, left: np.ndarray | None, right: np.ndarray | None) -> FrameReading:
        """`left`/`right`: landmarks (21, 3) in pixels, or None if the hand is not visible."""
        pressed: set[Button] = set()
        left_features = _features(left)
        right_features = _features(right)

        if right_features is not None:
            button = self._right_button(right_features)
            if button is not None:
                pressed.add(button)

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

    def _right_button(self, features: np.ndarray) -> Button | None:
        """The pressed right-thumb button (None = rest). One thumb: at most one button,
        the one the thumb has moved most clearly towards."""
        margins = {thumb.button: thumb.margin(features) for thumb in self._thumb_buttons}
        best = max(margins, key=margins.get)
        return best if margins[best] > 0 else None


def _features(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None or np.isnan(landmarks).any():
        return None
    features = thumb_position_features(landmarks)
    return None if np.isnan(features).any() else features


def _pooled_spread(*references: Reference) -> np.ndarray:
    """Common scale: root mean square of the poses' spreads."""
    return np.sqrt(np.mean([reference.spread ** 2 for reference in references], axis=0))
