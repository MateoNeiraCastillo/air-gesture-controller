"""Guided session: the script of actions and how to score what the controller did.

Two scripts:
- "classic": the Phase 2 exam. Identical blocks, so the start of the session
  can be compared with the end (does the posture drift over time?). It has no B,
  and keeps the same order as the 27-10 session (99.4 %) so results stay comparable.
- "data": for collecting a training dataset. Every button and combination,
  including B, in a random order per block (seeded), so a model can't learn the
  sequence instead of the gesture. The script is saved inside the session file,
  so each frame's label (what was asked) can be rebuilt later.

Scoring:
- "Hold" steps: frame-by-frame agreement of each button's CONFIRMED state
  (what the game would receive) against what was asked, ignoring the first
  `grace_ms` of the step (the player's reaction time).
- Steps with A taps: the confirmed A presses inside the step (shifted by the
  reaction margin) are counted against the requested ones.
"""

import random
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
    b_taps: int = 0                          # requested B taps (0 = B follows `hold`)


def _step(text: str, seconds: float, *hold: Button, a_taps: int = 0, b_taps: int = 0) -> Step:
    return Step(text, int(seconds * 1000), frozenset(hold), a_taps, b_taps)


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
    """The classic script: list of (block number, step)."""
    return [(block, step) for block in range(1, blocks + 1) for step in BLOCK]


# Data collection: every right-thumb state (rest, A, B) with every D-pad state
# (rest, left, right), plus taps, which give the model transitions to learn from.
DATA_ACTIONS = (
    _step("Hold LEFT", 3, Button.LEFT),
    _step("Hold RIGHT", 3, Button.RIGHT),
    _step("Hold A", 3, Button.A),
    _step("Hold B", 3, Button.B),
    _step("Hold LEFT and hold A", 3, Button.LEFT, Button.A),
    _step("Hold RIGHT and hold A", 3, Button.RIGHT, Button.A),
    _step("Hold LEFT and hold B", 3, Button.LEFT, Button.B),
    _step("Hold RIGHT and hold B", 3, Button.RIGHT, Button.B),
    _step("Press A 3 times", 4, a_taps=3),
    _step("Press B 3 times", 4, b_taps=3),
)
DATA_REST = _step("Rest", 2)


def build_data_script(blocks: int = 3, seed: int = 0) -> list[tuple[int, Step]]:
    """The data-collection script: each block has every action once, in a random
    order (reproducible with `seed`), with a rest after each action."""
    rng = random.Random(seed)
    script: list[tuple[int, Step]] = []
    for block in range(1, blocks + 1):
        actions = list(DATA_ACTIONS)
        rng.shuffle(actions)
        for action in actions:
            script += [(block, DATA_REST), (block, action)]
    return script


def script_arrays(script: list[tuple[int, Step]]) -> dict[str, np.ndarray]:
    """The script as arrays, to save it inside a session file."""
    return {
        "script_block": np.array([block for block, _ in script]),
        "script_text": np.array([step.text for _, step in script]),
        "script_duration_ms": np.array([step.duration_ms for _, step in script]),
        "script_hold_mask": np.array([to_mask(step.hold) for _, step in script]),
        "script_a_taps": np.array([step.a_taps for _, step in script]),
        "script_b_taps": np.array([step.b_taps for _, step in script]),
    }


def steps_from_session(data) -> list[tuple[int, Step]]:
    """The script a session was recorded with: saved in the file (data sessions),
    or the classic one (older sessions, which only saved the number of blocks)."""
    if "script_hold_mask" not in data:
        return build_script(int(data["blocks"]))
    return [(int(block), Step(str(text), int(duration), from_mask(int(mask)), int(a), int(b)))
            for block, text, duration, mask, a, b in zip(
                data["script_block"], data["script_text"], data["script_duration_ms"],
                data["script_hold_mask"], data["script_a_taps"], data["script_b_taps"])]


# In the classic script B is never asked for: it is scored as "must never be
# pressed" (ghost presses).
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
        if (button is Button.A and step.a_taps) or (button is Button.B and step.b_taps):
            agreement[button] = float("nan")       # taps are scored by counting
            continue
        expected = button in step.hold
        actual = (masks & bit) > 0
        agreement[button] = 100 * float(np.mean(actual == expected)) if len(masks) else float("nan")
    shifted = (press_times_a >= start_ms + 300) & (press_times_a < end_ms + 300)
    return StepScore(block, step, int(in_window.sum()), agreement, int(shifted.sum()))
