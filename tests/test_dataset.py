import numpy as np

from air_controller.buttons import Button
from air_controller.dataset import (
    LEFT_CLASSES,
    UNKNOWN,
    label_counts,
    session_frames,
    step_labels,
)
from air_controller.guided import (
    DATA_ACTIONS,
    GRACE_MS,
    Step,
    build_data_script,
    build_script,
    script_arrays,
    steps_from_session,
    to_mask,
)


def test_data_script_has_every_action_once_per_block_with_rests() -> None:
    script = build_data_script(blocks=3, seed=7)
    for block in (1, 2, 3):
        steps = [step for b, step in script if b == block]
        actions = [step for step in steps if step.text != "Rest"]
        assert sorted(a.text for a in actions) == sorted(a.text for a in DATA_ACTIONS)
        assert steps[0].text == "Rest"
    total_s = sum(step.duration_ms for _, step in script) / 1000
    assert 120 <= total_s <= 200


def test_data_script_order_is_random_but_reproducible() -> None:
    assert build_data_script(3, seed=1) == build_data_script(3, seed=1)
    assert build_data_script(3, seed=1) != build_data_script(3, seed=2)
    first, second = ([s.text for b, s in build_data_script(3, seed=1) if b == block] for block in (1, 2))
    assert first != second                       # blocks are shuffled independently


def test_data_script_asks_for_b() -> None:
    texts = [step.text for _, step in build_data_script(1, seed=0)]
    assert "Hold B" in texts and "Press B 3 times" in texts


def test_saved_script_round_trip() -> None:
    script = build_data_script(2, seed=3)
    assert steps_from_session(script_arrays(script)) == script


def test_old_sessions_fall_back_to_the_classic_script() -> None:
    assert steps_from_session({"blocks": np.array(2)}) == build_script(2)


def test_labels_ignore_the_reaction_time() -> None:
    step = Step("Hold LEFT and hold B", 3000, frozenset({Button.LEFT, Button.B}))
    assert step_labels(step, GRACE_MS - 1) == (UNKNOWN, UNKNOWN)
    assert step_labels(step, GRACE_MS) == (1, 2)


def test_labels_for_each_kind_of_step() -> None:
    assert step_labels(Step("Rest", 2000), 1000) == (0, 0)
    assert step_labels(Step("Hold RIGHT", 3000, frozenset({Button.RIGHT})), 1000) == (2, 0)
    assert step_labels(Step("Hold A", 3000, frozenset({Button.A})), 1000) == (0, 1)
    # taps: the right thumb moves at the player's rhythm, so it isn't labelled
    assert step_labels(Step("Press B 3 times", 4000, b_taps=3), 1000) == (0, UNKNOWN)
    assert step_labels(Step("Hold RIGHT and press A", 5000, frozenset({Button.RIGHT}), a_taps=3), 1000) == (2, UNKNOWN)


def test_session_frames_labels_each_frame_from_its_step() -> None:
    script = [(1, Step("Rest", 2000)), (1, Step("Hold A", 2000, frozenset({Button.A})))]
    timestamps = np.arange(0, 4000, 100)
    steps = (timestamps >= 2000).astype(int)
    hand = np.zeros((len(timestamps), 21, 3))
    data = {"timestamp_ms": timestamps, "step": steps, "step_starts_ms": np.array([0, 2000]),
            "left": hand, "right": hand, **script_arrays(script)}
    frames = session_frames(data)
    right = frames["right_label"]
    assert list(right[(timestamps >= GRACE_MS) & (timestamps < 2000)]) == [0] * 13
    assert all(right[(timestamps >= 2000) & (timestamps < 2000 + GRACE_MS)] == UNKNOWN)
    assert list(right[timestamps >= 2000 + GRACE_MS]) == [1] * 13
    assert frames["left"].dtype == np.float32


def test_label_counts() -> None:
    counts = label_counts(np.array([0, 0, 1, 2, -1]), LEFT_CLASSES)
    assert counts == {"rest": 2, "left": 1, "right": 1, "unknown": 1}


def test_hold_masks_are_saved_as_bits() -> None:
    arrays = script_arrays([(1, Step("Hold RIGHT and hold B", 3000, frozenset({Button.RIGHT, Button.B})))])
    assert arrays["script_hold_mask"][0] == to_mask({Button.RIGHT, Button.B})
