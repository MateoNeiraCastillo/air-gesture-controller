import numpy as np
import pytest

from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.guided import BUTTON_BITS
from air_controller.play_analysis import (
    a_b_switches,
    button_report,
    dpad_near_threshold_pct,
    hand_report,
    runs,
    slow_frames,
)
from air_controller.play_log import PlayLog
from air_controller.state_machine import ButtonEvent

A = BUTTON_BITS[Button.A]
B = BUTTON_BITS[Button.B]


def test_runs_finds_stretches() -> None:
    assert [(r.start, r.end) for r in runs(np.array([0, 1, 1, 0, 1]))] == [(1, 2), (4, 4)]
    assert runs(np.array([0, 0])) == []


def test_a_quick_tap_that_never_reached_the_game_is_a_lost_tap() -> None:
    t = np.arange(10) * 60
    raw = np.array([0, A, 0, 0, A, A, A, 0, 0, 0])          # one 1-frame tap, one real press
    held = np.array([0, 0, 0, 0, 0, A, A, A, 0, 0])
    events = (np.array([300, 420]), np.array(["a", "a"]), np.array([True, False]))
    report = button_report(Button.A, t, raw, held, *events)
    assert (report.raw_pulses, report.confirmed, report.lost_taps) == (2, 1, 1)
    assert report.median_delay_ms == 60                         # raw at 240, sent at 300


def test_short_presses_and_cut_holds_are_counted() -> None:
    t = np.arange(20) * 40
    raw = held = np.zeros(20, dtype=int)
    times = np.array([100, 150, 200, 600])                      # 50 ms press, then 50 ms gap
    report = button_report(Button.A, t, raw, held, times, np.array(["a"] * 4),
                           np.array([True, False, True, False]))
    assert report.short_presses == 1
    assert report.cut_holds == 1


def test_hand_report_measures_the_longest_gap() -> None:
    t = np.arange(8) * 50
    report = hand_report(t, np.array([1, 1, 0, 0, 0, 1, 0, 1], dtype=bool))
    assert report.losses == 2
    assert report.longest_gap_ms == 150
    assert report.visible_pct == 50


def test_dpad_near_threshold_ignores_frames_without_the_hand() -> None:
    left = np.array([0.25, 0.9, np.nan, 0.0])
    right = np.array([0.0, 0.0, np.nan, 0.0])
    assert dpad_near_threshold_pct(left, right, threshold=0.3) == pytest.approx(100 / 3)


def test_a_b_switches_and_slow_frames() -> None:
    assert a_b_switches(np.array([0, A, B, B, A, 0])) == 2
    assert slow_frames(np.array([0, 40, 80, 300, 340])) == 1


def test_play_log_round_trip(tmp_path) -> None:
    log = PlayLog()
    hand = np.zeros((21, 3))
    reading = FrameReading(frozenset({Button.A}), True, False, 0.1, 0.0)
    log.add(0, 40.0, hand, None, reading, frozenset(), [], [], paused=False)
    log.add(40, 41.0, hand, None, reading, frozenset({Button.A}), ["warn"],
            [ButtonEvent(Button.A, True, 40)], paused=False)
    log.add(80, 42.0, hand, None, reading, frozenset({Button.A}), [],
            [ButtonEvent(Button.A, False, 80)], paused=True)      # paused: not sent, not logged
    path = log.save(tmp_path / "play.npz", game="celeste", dpad_threshold=0.3)
    with np.load(path) as data:
        assert data["left"].shape == (3, 21, 3)
        assert np.isnan(data["right"]).all()
        assert list(data["raw_mask"]) == [A, A, A]
        assert list(data["held_mask"]) == [0, A, A]
        assert list(data["event_buttons"]) == ["a"] and list(data["event_pressed"]) == [True]
        assert list(data["posture_warning"]) == [False, True, False]
        assert str(data["game"]) == "celeste"
