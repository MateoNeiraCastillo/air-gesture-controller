"""ControllerLoop with fake frames, fake hands and a fake classifier: no camera, no MediaPipe."""

import time

import numpy as np
import pytest

from air_controller.buttons import Button
from air_controller.capture import Frame
from air_controller.classifier import FrameReading
from air_controller.config import TimingConfig
from air_controller.output import KeyOutput, RecordingKeySender
from air_controller.runtime import ControllerLoop
from air_controller.state_machine import ButtonStateMachine

KEYS = {Button.LEFT: "left", Button.RIGHT: "right", Button.A: "c", Button.B: "x"}
TIMING = TimingConfig(press_confirm_ms=40, release_confirm_ms=40, missing_hand_ms=200)
FRAME_MS = 40


class ListSource:
    """Frames whose image is just the index; the fake classifier reads `script[index]`."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.index = 0

    def read(self) -> Frame | None:
        if self.index >= self.count:
            return None
        frame = Frame(np.full((2, 2, 3), self.index, np.uint8), self.index * FRAME_MS, time.perf_counter())
        self.index += 1
        return frame


class ScriptedClassifier:
    """Returns the pressed buttons for each frame from a script (frame index → buttons)."""

    def __init__(self, script: list[frozenset[Button]]) -> None:
        self.script = script

    def classify(self, left, right, timestamp_ms: int) -> FrameReading:
        index = int(left[0, 0])
        return FrameReading(self.script[index], True, True, 0.0, 0.0)


class NoWarnings:
    def update(self, left, right, timestamp_ms: int) -> list[str]:
        return []


def detect(image: np.ndarray, timestamp_ms: int):
    hand = np.full((21, 3), float(image[0, 0, 0]))     # carries the frame index
    return hand, hand


def make_loop(script, sender: RecordingKeySender) -> tuple[ControllerLoop, ListSource]:
    source = ListSource(len(script))
    loop = ControllerLoop(source, detect, ScriptedClassifier(script), ButtonStateMachine(TIMING),
                          NoWarnings(), KeyOutput(sender, KEYS))
    return loop, source


def run_to_end(loop: ControllerLoop, timeout: float = 5.0) -> None:
    loop.start()
    deadline = time.perf_counter() + timeout
    while not loop.finished and time.perf_counter() < deadline:
        time.sleep(0.005)
    loop.stop()


A = frozenset({Button.A})
NONE: frozenset[Button] = frozenset()


def test_thread_sends_the_confirmed_presses_and_releases() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([NONE] * 2 + [A] * 4 + [NONE] * 4, sender)
    run_to_end(loop)
    assert sender.log == [("press", "c"), ("release", "c")]
    assert loop.finished


def test_stop_releases_a_key_held_when_the_frames_run_out() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([A] * 6, sender)
    run_to_end(loop)
    assert sender.log == [("press", "c"), ("release", "c")]


def test_snapshots_describe_the_latest_frame() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([NONE] * 2 + [A] * 4, sender)
    for _ in range(6):
        assert loop.step()
    snapshot = loop.wait_snapshot(timeout=0.1)
    assert snapshot.sequence == 6
    assert snapshot.held == {Button.A}
    assert snapshot.pressed_keys == {"c"}
    assert snapshot.presses == {Button.A: 1}
    assert loop.wait_snapshot(timeout=0.01) is None       # nothing new since


def test_pause_releases_and_sends_nothing_until_resumed() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([A] * 12, sender)
    for _ in range(4):
        loop.step()
    assert sender.log == [("press", "c")]
    loop.pause()
    for _ in range(4):
        loop.step()
    assert sender.log == [("press", "c"), ("release", "c")]
    assert loop.wait_snapshot(0.1).paused


def test_a_button_held_while_resuming_is_pressed_again() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([A] * 12, sender)
    loop.pause()
    for _ in range(4):
        loop.step()
    assert sender.log == []
    loop.resume()
    for _ in range(4):
        loop.step()
    assert sender.log == [("press", "c")]


def test_an_error_in_the_loop_stops_it_and_reaches_the_caller() -> None:
    def broken_detect(image, timestamp_ms):
        raise RuntimeError("camera exploded")

    sender = RecordingKeySender()
    source = ListSource(10)
    loop = ControllerLoop(source, broken_detect, ScriptedClassifier([NONE] * 10), ButtonStateMachine(TIMING),
                          NoWarnings(), KeyOutput(sender, KEYS))
    loop.start()
    deadline = time.perf_counter() + 2
    while not loop.finished and time.perf_counter() < deadline:
        time.sleep(0.005)
    assert loop.finished
    with pytest.raises(RuntimeError, match="camera exploded"):
        loop.stop()


def test_latency_is_measured_per_stage() -> None:
    sender = RecordingKeySender()
    loop, _ = make_loop([NONE] * 2 + [A] * 4, sender)
    run_to_end(loop)
    assert {"frame wait", "detection (MediaPipe)", "classify + states", "key sending"} <= set(loop.latency.stages)
    assert len(loop.latency.end_to_end_ms) == 1       # the press; the release happens in stop(), outside the loop


def test_the_loop_fills_the_play_log() -> None:
    from air_controller.play_log import PlayLog

    sender = RecordingKeySender()
    source = ListSource(8)
    log = PlayLog()
    script = [NONE] * 2 + [A] * 4 + [NONE] * 2
    loop = ControllerLoop(source, detect, ScriptedClassifier(script), ButtonStateMachine(TIMING),
                          NoWarnings(), KeyOutput(sender, KEYS), log=log)
    while loop.step():
        pass
    assert len(log) == 8
    assert [(b, pressed) for _, b, pressed in log.events] == [("a", True), ("a", False)]
