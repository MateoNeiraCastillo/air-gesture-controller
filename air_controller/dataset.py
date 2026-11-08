"""Turning guided sessions into a labelled dataset for training gesture models.

Each frame of a guided session gets two labels, one per hand, from what the
script was asking for at that moment:

- left hand (D-pad):   REST, LEFT or RIGHT
- right hand (thumb):  REST, A or B

A label is UNKNOWN (and the frame is left out of training for that hand) when
it can't be trusted:
- during the first `GRACE_MS` of every step, while the player is still reacting;
- for the right hand during tap steps ("Press A 3 times"), where the thumb goes
  up and down at the player's own rhythm.

Only hand landmarks are stored, never images: 21 points per hand.

Pure functions, no camera: tested in tests/test_dataset.py.
"""

from collections.abc import Mapping

import numpy as np

from air_controller.buttons import Button
from air_controller.guided import GRACE_MS, Step, steps_from_session

UNKNOWN = -1
REST = 0
LEFT_CLASSES = ("rest", "left", "right")     # label values 0, 1, 2 for the left hand
RIGHT_CLASSES = ("rest", "a", "b")           # label values 0, 1, 2 for the right hand


def step_labels(step: Step, ms_into_step: float, grace_ms: float = GRACE_MS) -> tuple[int, int]:
    """(left label, right label) for a frame `ms_into_step` after the step started."""
    if ms_into_step < grace_ms:
        return UNKNOWN, UNKNOWN
    if Button.LEFT in step.hold:
        left = 1
    elif Button.RIGHT in step.hold:
        left = 2
    else:
        left = REST
    if step.a_taps or step.b_taps:
        right = UNKNOWN
    elif Button.A in step.hold:
        right = 1
    elif Button.B in step.hold:
        right = 2
    else:
        right = REST
    return left, right


def session_frames(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Labels for every frame of one guided session (its own arrays, already loaded)."""
    script = steps_from_session(data)
    starts = np.asarray(data["step_starts_ms"])
    timestamps = np.asarray(data["timestamp_ms"])
    steps = np.asarray(data["step"])
    left_labels, right_labels = [], []
    for timestamp, index in zip(timestamps, steps):
        if index >= len(script) or index >= len(starts):
            left_labels.append(UNKNOWN)
            right_labels.append(UNKNOWN)
            continue
        left, right = step_labels(script[index][1], timestamp - starts[index])
        left_labels.append(left)
        right_labels.append(right)
    return {
        "timestamp_ms": timestamps,
        "step": steps,
        "block": np.array([script[i][0] if i < len(script) else 0 for i in steps]),
        "left": np.asarray(data["left"], dtype=np.float32),
        "right": np.asarray(data["right"], dtype=np.float32),
        "left_label": np.array(left_labels, dtype=np.int8),
        "right_label": np.array(right_labels, dtype=np.int8),
    }


def label_counts(labels: np.ndarray, classes: tuple[str, ...]) -> dict[str, int]:
    counts = {name: int(np.sum(labels == value)) for value, name in enumerate(classes)}
    counts["unknown"] = int(np.sum(labels == UNKNOWN))
    return counts
