"""The controller loop in its own thread: frame → hands → classifier → states → keys.

Why a thread: in play.py everything used to run in series (detect, send keys,
draw the dashboard, read the next frame). On 05-11 with Celeste the loop ran at
13 fps: ~42 ms of MediaPipe plus ~35 ms of drawing and showing the window. The
press confirmation needs two frames in a row, so every millisecond of drawing
was also added to every key press.

Now this loop only detects and sends keys, as fast as MediaPipe allows, and
publishes a `Snapshot` after each frame. The window (main thread, which macOS
requires for OpenCV windows) draws the latest snapshot at its own pace. Drawing
can no longer delay a key; at worst the dashboard shows a frame that is one
frame old.

Rules that keep it simple and safe:
- Only this thread touches the state machine, the posture monitor and the key
  output while it runs. The window asks for pause / resume through `pause()` and
  `resume()`, which are applied at the start of the next frame.
- `stop()` joins the thread and then releases every key. Never a stuck key.
- An exception in the loop stops it; `stop()` raises it in the caller.

No camera, MediaPipe or window code here: detection is injected as a function,
so the tests run with fake frames and fake hands.
"""

import queue
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from air_controller.buttons import Button
from air_controller.capture import Frame
from air_controller.classifier import FrameReading
from air_controller.latency import LatencyTracker
from air_controller.output import KeyOutput
from air_controller.play_log import PlayLog
from air_controller.state_machine import ButtonEvent, ButtonStateMachine

Hands = tuple[np.ndarray | None, np.ndarray | None]
DetectHands = Callable[[np.ndarray, int], Hands]


class FrameSource(Protocol):
    def read(self) -> Frame | None: ...


class HandClassifier(Protocol):
    def classify(self, left: np.ndarray | None, right: np.ndarray | None,
                 timestamp_ms: int) -> FrameReading: ...


class PostureCheck(Protocol):
    def update(self, left: np.ndarray | None, right: np.ndarray | None,
               timestamp_ms: int) -> list[str]: ...


@dataclass(frozen=True)
class Snapshot:
    """Everything the window needs to draw one frame. Immutable: safe to read from another thread."""

    sequence: int                     # increases by 1 per processed frame
    frame: Frame
    left: np.ndarray | None
    right: np.ndarray | None
    reading: FrameReading
    held: frozenset[Button]           # confirmed by the state machine
    pressed_keys: frozenset[str]      # keys currently down in the game
    warnings: tuple[str, ...]
    presses: dict[Button, int]        # confirmed presses since the start (a copy)
    paused: bool
    fps: float


class ControllerLoop:
    def __init__(self, source: FrameSource, detect: DetectHands, classifier: HandClassifier,
                 machine: ButtonStateMachine, posture: PostureCheck, output: KeyOutput,
                 latency: LatencyTracker | None = None, log: PlayLog | None = None) -> None:
        self._source = source
        self._detect = detect
        self._classifier = classifier
        self._machine = machine
        self._posture = posture
        self._output = output
        self.latency = latency or LatencyTracker()
        self.log = log                        # frame-by-frame record of the game, if given
        self._paused = False
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._presses: Counter[Button] = Counter()
        self._last_timestamp_ms = 0
        self._sequence = 0
        self._snapshot: Snapshot | None = None
        self._snapshot_lock = threading.Lock()
        self._new_snapshot = threading.Event()
        self._running = threading.Event()
        self._finished = threading.Event()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------------- control

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the loop, waits for it and releases every key. Re-raises a loop error."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._release_everything()
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def pause(self) -> None:
        self._commands.put("pause")

    def resume(self) -> None:
        self._commands.put("resume")

    @property
    def finished(self) -> bool:
        """True when the source ran out or the loop failed."""
        return self._finished.is_set()

    def wait_snapshot(self, timeout: float) -> Snapshot | None:
        """The newest snapshot once there is one the caller hasn't seen, or None after `timeout`."""
        if not self._new_snapshot.wait(timeout):
            return None
        with self._snapshot_lock:
            self._new_snapshot.clear()
            return self._snapshot

    # ------------------------------------------------------------------- loop

    def _run(self) -> None:
        try:
            while self._running.is_set() and self.step():
                pass
        except BaseException as error:  # noqa: BLE001  (handed to the caller in stop())
            self._error = error
        finally:
            self._finished.set()
            self._new_snapshot.set()          # wake up a window waiting for a frame

    def step(self) -> bool:
        """Processes one frame. False when the source has no more frames."""
        self._apply_commands()
        frame = self._source.read()
        if frame is None:
            return False
        self._last_timestamp_ms = frame.timestamp_ms
        start = time.perf_counter()
        self.latency.add_frame(start)
        self.latency.add_stage("frame wait", (start - frame.captured_s) * 1000)

        left, right = self._detect(frame.image, frame.timestamp_ms)
        detected = time.perf_counter()
        self.latency.add_stage("detection (MediaPipe)", (detected - start) * 1000)

        reading = self._classifier.classify(left, right, frame.timestamp_ms)
        events = self._machine.update(reading, frame.timestamp_ms)
        warnings = self._posture.update(left, right, frame.timestamp_ms)
        decided = time.perf_counter()
        self.latency.add_stage("classify + states", (decided - detected) * 1000)

        if events and not self._paused:
            self._send(events, frame, decided)
        if self.log is not None:
            self.log.add(frame.timestamp_ms, (detected - start) * 1000, left, right, reading,
                         self._machine.held, warnings, events, self._paused)
        self._publish(frame, left, right, reading, warnings)
        return True

    def _send(self, events: list[ButtonEvent], frame: Frame, decided: float) -> None:
        self._output.apply(events)
        self._presses.update(event.button for event in events if event.pressed)
        sent = time.perf_counter()
        self.latency.add_stage("key sending", (sent - decided) * 1000)
        self.latency.add_end_to_end((sent - frame.captured_s) * 1000)

    def _publish(self, frame: Frame, left, right, reading: FrameReading, warnings: list[str]) -> None:
        self._sequence += 1
        snapshot = Snapshot(
            sequence=self._sequence, frame=frame, left=left, right=right, reading=reading,
            held=self._machine.held, pressed_keys=self._output.pressed_keys,
            warnings=tuple(warnings), presses=dict(self._presses), paused=self._paused,
            fps=self.latency.fps(),
        )
        with self._snapshot_lock:
            self._snapshot = snapshot
            self._new_snapshot.set()

    def _apply_commands(self) -> None:
        while not self._commands.empty():
            command = self._commands.get_nowait()
            self._paused = command == "pause"
            # Pausing releases every key. Resuming starts from "nothing pressed" too, so a
            # button held at that moment is confirmed and sent again; otherwise its press
            # (which happened while paused) would never reach the game.
            self._release_everything()

    def _release_everything(self) -> None:
        self._machine.release_all(self._last_timestamp_ms)
        self._output.release_all()
