from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.config import TimingConfig
from air_controller.state_machine import ButtonEvent, ButtonStateMachine

TIMING = TimingConfig(press_confirm_ms=80, release_confirm_ms=80, missing_hand_ms=200)
FRAME_MS = 40


def reading(*pressed: Button, left: bool = True, right: bool = True) -> FrameReading:
    return FrameReading(frozenset(pressed), left, right, 0.0, 0.0)


def run(frames: list[FrameReading]) -> tuple[ButtonStateMachine, list[ButtonEvent]]:
    machine = ButtonStateMachine(TIMING)
    events = []
    for index, frame in enumerate(frames):
        events += machine.update(frame, index * FRAME_MS)
    return machine, events


def test_single_frame_blip_is_ignored() -> None:
    _, events = run([reading(), reading(Button.A), reading(), reading()])
    assert events == []


def test_sustained_press_is_confirmed_after_press_confirm_ms() -> None:
    _, events = run([reading()] + [reading(Button.A)] * 4)
    # seen pressed from t=40; confirmed 80 ms later (t=120)
    assert events == [ButtonEvent(Button.A, True, 120)]


def test_release_needs_release_confirm_ms() -> None:
    frames = [reading(Button.A)] * 4 + [reading()] * 4
    machine, events = run(frames)
    assert events == [ButtonEvent(Button.A, True, 80), ButtonEvent(Button.A, False, 240)]
    assert machine.held == frozenset()


def test_one_frame_gap_while_holding_does_not_release() -> None:
    frames = [reading(Button.RIGHT)] * 4 + [reading()] + [reading(Button.RIGHT)] * 4
    machine, events = run(frames)
    assert events == [ButtonEvent(Button.RIGHT, True, 80)]
    assert machine.held == {Button.RIGHT}


def test_short_hand_loss_keeps_button_held() -> None:
    frames = [reading(Button.A)] * 4 + [reading(right=False)] * 3 + [reading(Button.A)] * 2
    machine, _ = run(frames)
    assert machine.held == {Button.A}


def test_long_hand_loss_releases_only_that_hands_buttons() -> None:
    frames = [reading(Button.A, Button.LEFT)] * 4 + [reading(Button.LEFT, right=False)] * 8
    machine, events = run(frames)
    assert ButtonEvent(Button.A, False, 360) in events      # lost since t=160, +200 ms
    assert machine.held == {Button.LEFT}


def test_missing_hand_never_confirms_a_press() -> None:
    _, events = run([reading(Button.A)] + [reading(right=False)] * 10)
    assert events == []


def test_dpad_and_a_are_independent() -> None:
    frames = [reading(Button.RIGHT)] * 3 + [reading(Button.RIGHT, Button.A)] * 4
    machine, events = run(frames)
    assert [e.button for e in events] == [Button.RIGHT, Button.A]
    assert machine.held == {Button.RIGHT, Button.A}


def test_release_all_releases_everything_once() -> None:
    machine, _ = run([reading(Button.A, Button.LEFT)] * 4)
    released = machine.release_all(1000)
    assert {e.button for e in released} == {Button.A, Button.LEFT}
    assert all(not e.pressed for e in released)
    assert machine.release_all(1040) == []
