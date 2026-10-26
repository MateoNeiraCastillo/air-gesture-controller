import math

import numpy as np

from air_controller.buttons import Button
from air_controller.guided import (
    GRACE_MS,
    Step,
    build_script,
    from_mask,
    score_step,
    to_mask,
)

FRAME_MS = 40


def test_script_repeats_identical_blocks() -> None:
    script = build_script(blocks=3)
    per_block = [[step for block, step in script if block == b] for b in (1, 2, 3)]
    assert per_block[0] == per_block[1] == per_block[2]
    total_s = sum(step.duration_ms for _, step in script) / 1000
    assert 100 <= total_s <= 180


def test_mask_round_trip() -> None:
    buttons = frozenset({Button.RIGHT, Button.A})
    assert from_mask(to_mask(buttons)) == buttons


def frames(duration_ms: int, mask_fn) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(0, duration_ms, FRAME_MS)
    return timestamps, np.array([mask_fn(t) for t in timestamps])


def test_perfect_hold_scores_100_and_ignores_reaction_time() -> None:
    step = Step("Manten DERECHA", 4000, frozenset({Button.RIGHT}))
    # el jugador tarda 500 ms en reaccionar: no debe penalizar
    timestamps, masks = frames(4000, lambda t: to_mask({Button.RIGHT}) if t >= 500 else 0)
    score = score_step(1, step, 0, timestamps, masks, np.array([]))
    assert score.agreement[Button.RIGHT] == 100.0
    assert score.agreement[Button.LEFT] == 100.0
    assert score.frames_scored == len([t for t in timestamps if t >= GRACE_MS])


def test_ghost_press_lowers_agreement() -> None:
    step = Step("Reposo", 2000)
    timestamps, masks = frames(2000, lambda t: to_mask({Button.A}) if 1000 <= t < 1500 else 0)
    score = score_step(1, step, 0, timestamps, masks, np.array([]))
    assert score.agreement[Button.A] < 100.0
    assert score.agreement[Button.LEFT] == 100.0


def test_taps_are_counted_not_scored_frame_by_frame() -> None:
    step = Step("Pulsa A 3 veces", 4000, a_taps=3)
    timestamps, masks = frames(4000, lambda t: 0)
    presses = np.array([900, 1900, 2900, 5000])          # la última es de otro paso
    score = score_step(1, step, 0, timestamps, masks, presses)
    assert score.a_presses == 3
    assert math.isnan(score.agreement[Button.A])
