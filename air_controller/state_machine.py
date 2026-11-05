"""Layer 4: from per-frame readings to real presses.

The classifier says what looks pressed in each frame, with the occasional
flicker. The state machine decides when to actually press and release:

- Press: the button must be seen pressed for `press_confirm_ms` in a row.
- Release: it must be seen released for `release_confirm_ms` in a row.
- Hand not visible: means "no information", not "released". The button keeps
  its state; if the hand is missing for more than `missing_hand_ms`, its
  buttons are released so no keys are left stuck.

It works with times in milliseconds, not frames: on CPU we run at ~25 fps,
not 30, and the frame rate can vary.
"""

from dataclasses import dataclass, field

from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.config import TimingConfig

# Which hand controls each button.
HAND_OF = {Button.LEFT: "left", Button.RIGHT: "left", Button.A: "right"}


@dataclass(frozen=True)
class ButtonEvent:
    button: Button
    pressed: bool          # True = press, False = release
    timestamp_ms: int


@dataclass
class _ButtonState:
    pressed: bool = False
    change_since_ms: int | None = None    # since when the opposite of the current state has been seen


@dataclass
class ButtonStateMachine:
    timing: TimingConfig
    _buttons: dict[Button, _ButtonState] = field(
        default_factory=lambda: {button: _ButtonState() for button in Button})
    _missing_since_ms: dict[str, int | None] = field(
        default_factory=lambda: {"left": None, "right": None})

    @property
    def held(self) -> frozenset[Button]:
        return frozenset(button for button, state in self._buttons.items() if state.pressed)

    def update(self, reading: FrameReading, timestamp_ms: int) -> list[ButtonEvent]:
        """Processes one frame's reading. Returns the events to send."""
        visible = {"left": reading.left_visible, "right": reading.right_visible}
        self._track_missing(visible, timestamp_ms)
        events: list[ButtonEvent] = []
        for button, state in self._buttons.items():
            hand = HAND_OF[button]
            if visible[hand]:
                event = self._observe(button, state, button in reading.pressed, timestamp_ms)
            else:
                event = self._without_hand(button, state, hand, timestamp_ms)
            if event is not None:
                events.append(event)
        return events

    def release_all(self, timestamp_ms: int) -> list[ButtonEvent]:
        """Releases everything (on exit or recalibration): never leave a key stuck."""
        events = []
        for button, state in self._buttons.items():
            if state.pressed:
                events.append(ButtonEvent(button, False, timestamp_ms))
            state.pressed = False
            state.change_since_ms = None
        return events

    # ------------------------------------------------------------------ #

    def _observe(self, button: Button, state: _ButtonState, seen_pressed: bool,
                 now: int) -> ButtonEvent | None:
        if seen_pressed == state.pressed:
            state.change_since_ms = None          # matches the state: nothing pending
            return None
        if state.change_since_ms is None:
            state.change_since_ms = now           # the change starts being seen
        confirm = self.timing.press_confirm_ms if seen_pressed else self.timing.release_confirm_ms
        if now - state.change_since_ms < confirm:
            return None
        state.pressed = seen_pressed
        state.change_since_ms = None
        return ButtonEvent(button, seen_pressed, now)

    def _without_hand(self, button: Button, state: _ButtonState, hand: str,
                      now: int) -> ButtonEvent | None:
        state.change_since_ms = None              # no information: confirms nothing
        missing_since = self._missing_since_ms[hand]
        if state.pressed and missing_since is not None and \
                now - missing_since >= self.timing.missing_hand_ms:
            state.pressed = False
            return ButtonEvent(button, False, now)
        return None

    def _track_missing(self, visible: dict[str, bool], now: int) -> None:
        for hand, is_visible in visible.items():
            if is_visible:
                self._missing_since_ms[hand] = None
            elif self._missing_since_ms[hand] is None:
                self._missing_since_ms[hand] = now
