"""A record of a real game: what the camera saw and what the controller did, frame by frame.

Guided sessions measure accuracy against a script. A real game has no script,
so when "it doesn't detect things well" the only way to find out what went
wrong is to look at the frames: was the hand lost, was the thumb hovering near
the threshold, was a quick tap too short to be confirmed?

The file uses the same field names as a guided session (`left`, `right`,
`raw_mask`, `held_mask`, `event_*`...), so the same tools can read both, and
the landmarks can be replayed offline with a different classifier or timing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.guided import to_mask
from air_controller.state_machine import ButtonEvent

NO_HAND = np.full((21, 3), np.nan)


@dataclass
class PlayLog:
    timestamp_ms: list[int] = field(default_factory=list)
    detection_ms: list[float] = field(default_factory=list)
    left: list[np.ndarray] = field(default_factory=list)
    right: list[np.ndarray] = field(default_factory=list)
    left_visible: list[bool] = field(default_factory=list)
    right_visible: list[bool] = field(default_factory=list)
    raw_mask: list[int] = field(default_factory=list)       # what the classifier saw in the frame
    held_mask: list[int] = field(default_factory=list)      # confirmed by the state machine
    dpad_left_progress: list[float] = field(default_factory=list)
    dpad_right_progress: list[float] = field(default_factory=list)
    posture_warning: list[bool] = field(default_factory=list)
    paused: list[bool] = field(default_factory=list)
    events: list[tuple[int, str, bool]] = field(default_factory=list)   # (ms, button, pressed)

    def add(self, timestamp_ms: int, detection_ms: float, left: np.ndarray | None,
            right: np.ndarray | None, reading: FrameReading, held: frozenset[Button],
            warnings: list[str], events: list[ButtonEvent], paused: bool) -> None:
        self.timestamp_ms.append(timestamp_ms)
        self.detection_ms.append(detection_ms)
        self.left.append(NO_HAND if left is None else left)
        self.right.append(NO_HAND if right is None else right)
        self.left_visible.append(reading.left_visible)
        self.right_visible.append(reading.right_visible)
        self.raw_mask.append(to_mask(reading.pressed))
        self.held_mask.append(to_mask(held))
        self.dpad_left_progress.append(reading.dpad_left_progress)
        self.dpad_right_progress.append(reading.dpad_right_progress)
        self.posture_warning.append(bool(warnings))
        self.paused.append(paused)
        if not paused:
            self.events.extend((event.timestamp_ms, event.button.value, event.pressed) for event in events)

    def __len__(self) -> int:
        return len(self.timestamp_ms)

    def save(self, path: Path, **metadata) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            timestamp_ms=np.array(self.timestamp_ms, dtype=np.int64),
            detection_ms=np.array(self.detection_ms),
            left=np.array(self.left).reshape(-1, 21, 3),
            right=np.array(self.right).reshape(-1, 21, 3),
            left_visible=np.array(self.left_visible, dtype=bool),
            right_visible=np.array(self.right_visible, dtype=bool),
            raw_mask=np.array(self.raw_mask, dtype=np.int64),
            held_mask=np.array(self.held_mask, dtype=np.int64),
            dpad_left_progress=np.array(self.dpad_left_progress),
            dpad_right_progress=np.array(self.dpad_right_progress),
            posture_warning=np.array(self.posture_warning, dtype=bool),
            paused=np.array(self.paused, dtype=bool),
            event_times_ms=np.array([t for t, _, _ in self.events], dtype=np.int64),
            event_buttons=np.array([b for _, b, _ in self.events], dtype="<U5"),
            event_pressed=np.array([p for _, _, p in self.events], dtype=bool),
            **{key: np.array(value) for key, value in metadata.items()},
        )
        return path
