import pytest

from air_controller.buttons import Button
from air_controller.latency import LatencyTracker
from air_controller.output import KeyOutput, RecordingKeySender, validate_key_name
from air_controller.state_machine import ButtonEvent

KEYS = {Button.LEFT: "left", Button.RIGHT: "right", Button.A: "up"}


def press(button: Button, t: int = 0) -> ButtonEvent:
    return ButtonEvent(button, True, t)


def release(button: Button, t: int = 0) -> ButtonEvent:
    return ButtonEvent(button, False, t)


def test_events_become_key_presses_and_releases() -> None:
    sender = RecordingKeySender()
    output = KeyOutput(sender, KEYS)
    output.apply([press(Button.RIGHT), press(Button.A), release(Button.A)])
    assert sender.log == [("press", "right"), ("press", "up"), ("release", "up")]
    assert output.pressed_keys == {"right"}


def test_duplicate_events_do_not_double_press() -> None:
    sender = RecordingKeySender()
    output = KeyOutput(sender, KEYS)
    output.apply([press(Button.A), press(Button.A), release(Button.A), release(Button.A)])
    assert sender.log == [("press", "up"), ("release", "up")]


def test_release_all_frees_every_held_key() -> None:
    sender = RecordingKeySender()
    output = KeyOutput(sender, KEYS)
    output.apply([press(Button.LEFT), press(Button.A)])
    output.release_all()
    assert ("release", "left") in sender.log and ("release", "up") in sender.log
    assert output.pressed_keys == frozenset()


def test_context_manager_releases_even_on_error() -> None:
    sender = RecordingKeySender()
    with pytest.raises(RuntimeError):
        with KeyOutput(sender, KEYS) as output:
            output.apply([press(Button.RIGHT)])
            raise RuntimeError("failure in the middle of the game")
    assert sender.log[-1] == ("release", "right")


@pytest.mark.parametrize("key", ["left", "space", "z", "x"])
def test_valid_key_names(key: str) -> None:
    validate_key_name(key)


@pytest.mark.parametrize("key", ["arrow_left", "", "ab"])
def test_invalid_key_names_fail_at_startup(key: str) -> None:
    with pytest.raises(ValueError):
        KeyOutput(RecordingKeySender(), {**KEYS, Button.A: key})


def test_latency_summary() -> None:
    tracker = LatencyTracker()
    for i in range(10):
        tracker.add_stage("detection", 30 + i)
        tracker.add_frame(i * 0.04)
    tracker.add_end_to_end(50.0)
    text = tracker.summary()
    assert "detection" in text and "frame → key" in text
    assert tracker.fps() == pytest.approx(25.0)
