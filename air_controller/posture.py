"""Posture monitor: warns when the hand is no longer as it was during calibration.

In the spike, the daytime `rhythm` clip failed because the fist rotated to face
the camera: the hand scale (wrist → middle finger base) dropped from ~60 px to
1–29 px and the thumb measurements blew up. We don't fix it in the
classifier; we DETECT it: if a hand's apparent size leaves the range seen
during calibration (widened by the config margins) for `warning_after_ms`
in a row, a warning tells the player to reposition.

It compares against a RANGE rather than a single size because the hand tilts
somewhat when moving the thumb: in the daytime spike, the left hand measured
from 77 % to 130 % of its median depending on the pose, all of them valid.
"""

from dataclasses import dataclass, field

import numpy as np

from air_controller.calibration import Calibration
from air_controller.config import PostureConfig
from air_controller.features import hand_size

HAND_NAMES = {"left": "Left", "right": "Right"}


@dataclass
class PostureMonitor:
    calibration: Calibration
    config: PostureConfig
    _out_since_ms: dict[str, int | None] = field(
        default_factory=lambda: {"left": None, "right": None})
    _last_ratio: dict[str, float] = field(
        default_factory=lambda: {"left": float("nan"), "right": float("nan")})

    def update(self, left: np.ndarray | None, right: np.ndarray | None,
               timestamp_ms: int) -> list[str]:
        """Returns the active warnings (empty list if everything is fine)."""
        warnings = []
        for hand, landmarks in (("left", left), ("right", right)):
            ratio = self._ratio(hand, landmarks)
            self._last_ratio[hand] = ratio
            low, high = self._limits(hand)
            size = ratio * self._reference(hand)
            out_of_range = not np.isnan(ratio) and not (low <= size <= high)
            if not out_of_range:
                self._out_since_ms[hand] = None
                continue
            if self._out_since_ms[hand] is None:
                self._out_since_ms[hand] = timestamp_ms
            if timestamp_ms - self._out_since_ms[hand] >= self.config.warning_after_ms:
                warnings.append(self._message(hand, ratio))
        return warnings

    def ratio(self, hand: str) -> float:
        """Last measured relative size (1.0 = as during calibration)."""
        return self._last_ratio[hand]

    def _reference(self, hand: str) -> float:
        return (self.calibration.left_hand_size if hand == "left"
                else self.calibration.right_hand_size)

    def _limits(self, hand: str) -> tuple[float, float]:
        """Minimum and maximum acceptable size, in pixels."""
        smallest, largest = (self.calibration.left_size_range if hand == "left"
                             else self.calibration.right_size_range)
        if largest <= 0:   # calibration without a range: use the median
            smallest = largest = self._reference(hand)
        return (smallest * self.config.hand_size_ratio_min, largest * self.config.hand_size_ratio_max)

    def _ratio(self, hand: str, landmarks: np.ndarray | None) -> float:
        if landmarks is None or np.isnan(landmarks).any():
            return float("nan")
        return hand_size(landmarks) / self._reference(hand)

    def _message(self, hand: str, ratio: float) -> str:
        name = HAND_NAMES[hand]
        if ratio * self._reference(hand) < self._limits(hand)[0]:
            return (f"{name} hand: rotated or moved away ({ratio:.0%} of the calibrated size). "
                    "Return to the calibration posture.")
        return (f"{name} hand: too close ({ratio:.0%} of the calibrated size). "
                "Return to the calibration posture.")
