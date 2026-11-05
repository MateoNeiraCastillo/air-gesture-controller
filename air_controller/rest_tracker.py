"""Re-learning where the thumb rests while you play.

What two real games showed (05-11, Celeste): the right thumb's resting position
crept towards "A pressed". Measured along rest (0) → calibrated press (1), the
most relaxed 20 % of the frames went from ~0.2 to 0.6–1.0, and A was "held"
81–94 % of the time in some windows: the character jumped by itself. In the
first game the fist rolled from −4° to −20° (the thumb is measured on screen
axes, so rolling looks like the thumb going down); in the second it didn't, the
relaxed thumb simply sat lower than the straight thumb held while calibrating.
Real presses went deeper than the calibrated one (up to 1.2–2.8), so they can
still be told apart from that new rest.

So the controller keeps re-learning the rest of button A: it
takes a low percentile of its position over the last few seconds. Presses are
short compared with the window, so that percentile is where the thumb is when
it is NOT pressing, and a button is pressed when the thumb goes
`button_threshold` past it. Replayed on both games (5 s, 20th percentile, rest
capped at 0.8): A held 23–39 % and 11–47 % per window instead of up to 86 % and 98 %.

Limits that keep it safe:
- The rest can move at most `max_shift` of the way to the calibrated press (0.8),
  so a press `button_threshold` deeper than that always counts.
- Until the history covers `MIN_SPAN` of the window (4 of the 5 s), the calibrated
  rest is used. Learning from a short history was tested and rejected: a press
  held from the start (the spike's `a_held` clip) became "the rest" after 10
  frames and was released after 0.3 s.
- Holding a button for longer than the window slowly raises the rest, so a
  long, light hold may end up released. Games where you hold a button for
  seconds (accelerating in a racing game) should turn it off: rest_window_ms = 0.
- Only for A (see RELEARNED). The D-pad drifted much less in that game (0.07 → 0.22, with
  its threshold at 0.3), and holding it for seconds to walk is normal.
"""

from collections import deque
from collections.abc import Mapping

import numpy as np

from air_controller.buttons import Button
from air_controller.classifier import Classifier, FrameReading
from air_controller.config import ClassifierConfig

MIN_SAMPLES = 10
MIN_SPAN = 0.8          # fraction of the window the history must cover


class RestTracker:
    def __init__(self, window_ms: float, percentile: float, max_shift: float) -> None:
        self.window_ms = window_ms
        self.percentile = percentile
        self.max_shift = max_shift
        self._history: dict[Button, deque[tuple[int, float]]] = {}

    @classmethod
    def from_config(cls, config: ClassifierConfig) -> "RestTracker":
        return cls(config.rest_window_ms, config.rest_percentile, config.rest_max_shift)

    @property
    def enabled(self) -> bool:
        return self.window_ms > 0

    def update(self, timestamp_ms: int, progress: Mapping[Button, float]) -> None:
        if not self.enabled:
            return
        for button, value in progress.items():
            history = self._history.setdefault(button, deque())
            history.append((timestamp_ms, value))
            while history and timestamp_ms - history[0][0] > self.window_ms:
                history.popleft()

    def offsets(self) -> dict[Button, float]:
        """Current rest of each button, as progress from the calibrated rest (0 = unchanged)."""
        out = {}
        for button, history in self._history.items():
            span = history[-1][0] - history[0][0] if history else 0
            if len(history) >= MIN_SAMPLES and span >= MIN_SPAN * self.window_ms:
                low = float(np.percentile([value for _, value in history], self.percentile))
                out[button] = min(max(low, 0.0), self.max_shift)
        return out


# Only A's rest is re-learned. Re-learning B's too was tested and rejected: in the
# second 05-11 game B's learned rest rose to ~0.5 and 28 % of the outward tilts
# were then read as A or nothing; with B on its calibrated rest, 100 % were B.
RELEARNED = (Button.A,)


class AdaptiveClassifier:
    """The classifier plus a RestTracker: each frame is judged from the rest learned
    on the previous frames, then added to what is learned (button A only)."""

    def __init__(self, classifier: Classifier, tracker: RestTracker) -> None:
        self.classifier = classifier
        self.tracker = tracker

    def classify(self, left: np.ndarray | None, right: np.ndarray | None, timestamp_ms: int) -> FrameReading:
        reading = self.classifier.classify(left, right, self.tracker.offsets())
        self.tracker.update(timestamp_ms, {button: value for button, value in reading.thumb_progress.items()
                                           if button in RELEARNED})
        return reading
