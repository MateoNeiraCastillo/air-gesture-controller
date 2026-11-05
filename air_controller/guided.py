"""Guided session: the script of actions and how to score what the controller did.

The script repeats in identical blocks so the start of the session can be
compared with the end (does the posture drift over time?).

Scoring:
- "Hold" steps: frame-by-frame agreement of each button's CONFIRMED state
  (what the game would receive) against what was asked, ignoring the first
  `grace_ms` of the step (the player's reaction time).
- Steps with A taps: the confirmed A presses inside the step (shifted by the
  reaction margin) are counted against the requested ones.
"""

from dataclasses import dataclass

import numpy as np

from air_controller.buttons import Button

GRACE_MS = 700


@dataclass(frozen=True)
class Step:
    text: str
    duration_ms: int
    hold: frozenset[Button] = frozenset()   # buttons that must be held
    a_taps: int = 0                          # requested A taps (0 = A follows `hold`)


def _step(text: str, seconds: float, *hold: Button, a_taps: int = 0) -> Step:
    return Step(text, int(seconds * 1000), frozenset(hold), a_taps)


BLOCK = (
    _step("Rest", 4),
    _step("Hold RIGHT", 4, Button.RIGHT),
    _step("Rest", 3),
    _step("Hold LEFT", 4, Button.LEFT),
    _step("Rest", 3),
    _step("Hold A", 3, Button.A),
    _step("Rest", 3),
    _step("Press A 3 times", 4, a_taps=3),
    _step("Rest", 2),
    _step("Hold RIGHT and press A 3 times", 5, Button.RIGHT, a_taps=3),
    _step("Rest", 3),
    _step("Hold LEFT and press A 2 times", 5, Button.LEFT, a_taps=2),
    _step("Rest", 2),
    _step("Hold RIGHT and hold A", 4, Button.RIGHT, Button.A),
)


def build_script(blocks: int = 3) -> list[tuple[int, Step]]:
    """List of (block number, step)."""
    return [(block, step) for block in range(1, blocks + 1) for step in BLOCK]


# B is not in the script: it is scored as "must never be pressed" (ghost presses).
BUTTON_BITS = {Button.LEFT: 1, Button.RIGHT: 2, Button.A: 4, Button.B: 8}


def to_mask(buttons: frozenset[Button] | set[Button]) -> int:
    return sum(BUTTON_BITS[button] for button in buttons)


def from_mask(mask: int) -> frozenset[Button]:
    return frozenset(button for button, bit in BUTTON_BITS.items() if mask & bit)


@dataclass(frozen=True)
class StepScore:
    block: int
    step: Step
    frames_scored: int
    agreement: dict[Button, float]     # % of frames with the correct state (NaN if not scored)
    a_presses: int                     # confirmed A presses during the step


def score_step(block: int, step: Step, start_ms: int, timestamps: np.ndarray,
               held_masks: np.ndarray, press_times_a: np.ndarray) -> StepScore:
    """Scores a step from the session's frames.

    `held_masks`: confirmed state per frame (bit mask).
    `press_times_a`: times (ms) of the confirmed A presses.
    """
    end_ms = start_ms + step.duration_ms
    in_window = (timestamps >= start_ms + GRACE_MS) & (timestamps < end_ms)
    masks = held_masks[in_window]
    agreement: dict[Button, float] = {}
    for button, bit in BUTTON_BITS.items():
        if button is Button.A and step.a_taps:
            agreement[button] = float("nan")       # taps are scored by counting
            continue
        expected = button in step.hold
        actual = (masks & bit) > 0
        agreement[button] = 100 * float(np.mean(actual == expected)) if len(masks) else float("nan")
    shifted = (press_times_a >= start_ms + 300) & (press_times_a < end_ms + 300)
    return StepScore(block, step, int(in_window.sum()), agreement, int(shifted.sum()))
