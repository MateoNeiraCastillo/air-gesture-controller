"""Layer 5: turn button events into key presses and releases.

Golden rule: NEVER leave a key stuck. A stuck key in a game is worse than a
missed press (Mario runs off a cliff on his own). Therefore:
- Each sender remembers which keys it has pressed and `release_all()` releases them.
- `KeyOutput` is a context manager: on leaving the `with` (normally, through
  an exception or with Ctrl+C) it releases everything.
- In addition, the state machine releases the buttons of a hand that has not
  been seen for > `missing_hand_ms`: moving your hands away from the camera is
  a natural "emergency brake".

The real sender uses pynput, which is only imported when it is created: the
tests use `RecordingKeySender` and need neither pynput nor system permissions.
"""

from dataclasses import dataclass, field
from typing import Protocol

from air_controller.buttons import Button
from air_controller.state_machine import ButtonEvent

SPECIAL_KEYS = {"left", "right", "up", "down", "space", "enter", "shift", "ctrl", "alt",
                "tab", "esc"}


class KeySender(Protocol):
    def press(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...


def validate_key_name(key: str) -> None:
    """Accepts special keys (left, space...) or a single character (z, x...)."""
    if key not in SPECIAL_KEYS and len(key) != 1:
        raise ValueError(f"Unknown key: '{key}'. Use one of {sorted(SPECIAL_KEYS)} "
                         "or a single character.")


class PynputKeySender:
    """Sends real keys to the system.

    On macOS the app it runs from (Terminal or VS Code) needs the Accessibility
    permission; without it, keys are silently ignored.
    """

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key, KeyCode  # lazy import

        self._controller = Controller()
        self._special = {name: getattr(Key, name) for name in SPECIAL_KEYS}
        self._key_code = KeyCode

    def _resolve(self, key: str):
        return self._special.get(key) or self._key_code.from_char(key)

    def press(self, key: str) -> None:
        self._controller.press(self._resolve(key))

    def release(self, key: str) -> None:
        self._controller.release(self._resolve(key))


@dataclass
class RecordingKeySender:
    """Fake sender for tests and --dry-run: records what would have been sent."""

    log: list[tuple[str, str]] = field(default_factory=list)   # (action, key)
    echo: bool = False

    def press(self, key: str) -> None:
        self.log.append(("press", key))
        if self.echo:
            print(f"  [key] ↓ {key}")

    def release(self, key: str) -> None:
        self.log.append(("release", key))
        if self.echo:
            print(f"  [key] ↑ {key}")


class KeyOutput:
    """Translates ButtonEvent → keys, remembering what is pressed so everything can be released."""

    def __init__(self, sender: KeySender, keys: dict[Button, str]) -> None:
        for key in keys.values():
            validate_key_name(key)
        self._sender = sender
        self._keys = keys
        self._down: set[str] = set()

    def apply(self, events: list[ButtonEvent]) -> None:
        for event in events:
            key = self._keys[event.button]
            if event.pressed and key not in self._down:
                self._sender.press(key)
                self._down.add(key)
            elif not event.pressed and key in self._down:
                self._sender.release(key)
                self._down.discard(key)

    def release_all(self) -> None:
        for key in sorted(self._down):
            self._sender.release(key)
        self._down.clear()

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._down)

    def __enter__(self) -> "KeyOutput":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release_all()
