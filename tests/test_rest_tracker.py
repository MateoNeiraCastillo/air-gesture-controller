import pytest

from air_controller.buttons import Button
from air_controller.calibration import CalibrationCollector
from air_controller.rest_tracker import MIN_SAMPLES, AdaptiveClassifier, RestTracker
from test_classifier import B_PRESSED, PRESSED, REST, between, build_classifier, make_hand  # same directory


def feed(tracker: RestTracker, values: list[float], start_ms: int = 0, step_ms: int = 50) -> None:
    for index, value in enumerate(values):
        tracker.update(start_ms + index * step_ms, {Button.A: value})


def test_disabled_tracker_never_moves_the_rest() -> None:
    tracker = RestTracker(window_ms=0, percentile=20, max_shift=0.5)
    feed(tracker, [0.4] * 50)
    assert tracker.offsets() == {}


def test_calibrated_rest_until_there_is_enough_history() -> None:
    tracker = RestTracker(5000, 20, 0.5)
    feed(tracker, [0.4] * (MIN_SAMPLES - 1))
    assert tracker.offsets() == {}
    feed(tracker, [0.4] * 70)                        # 3.5 s: still under 4 s of history
    assert tracker.offsets() == {}
    feed(tracker, [0.4] * 20, start_ms=3500)          # now 4.45 s
    assert tracker.offsets()[Button.A] == pytest.approx(0.4)


def test_rest_is_a_low_percentile_so_presses_dont_count_as_rest() -> None:
    tracker = RestTracker(5000, 20, 0.5)
    feed(tracker, [0.3] * 70 + [1.0] * 30)        # resting at 0.3, pressing 30 % of the time
    assert tracker.offsets()[Button.A] == pytest.approx(0.3)


def test_old_frames_leave_the_window() -> None:
    tracker = RestTracker(1000, 20, 0.5)
    feed(tracker, [0.4] * 40)                       # 0–2 s
    feed(tracker, [0.0] * 40, start_ms=3000)        # 3–5 s
    assert tracker.offsets()[Button.A] == pytest.approx(0.0)


def test_the_rest_never_moves_more_than_max_shift() -> None:
    tracker = RestTracker(5000, 20, 0.5)
    feed(tracker, [0.9] * 100)                      # held for 5 s
    assert tracker.offsets()[Button.A] == 0.5


def make_adaptive(window_ms: float = 5000) -> AdaptiveClassifier:
    """Like config.toml: pressed 0.4 past the rest, rest capped at 0.8."""
    classifier = build_classifier(CalibrationCollector(frames_per_pose=60), button_threshold=0.4)
    return AdaptiveClassifier(classifier, RestTracker(window_ms, 20, 0.8))


def test_a_thumb_that_rests_lower_than_calibrated_stops_pressing_a() -> None:
    """The 05-11 Celeste case: the relaxed thumb sits 60 % of the way to A."""
    drifted = make_hand(*between(REST, PRESSED, 0.6), noise=0)
    left = make_hand(*REST)
    fixed = make_adaptive(window_ms=0)
    adaptive = make_adaptive()
    for frame in range(120):                        # 6 s at 20 fps
        fixed_reading = fixed.classify(left, drifted, frame * 50)
        adaptive_reading = adaptive.classify(left, drifted, frame * 50)
    assert Button.A in fixed_reading.pressed        # without re-learning: A stuck on
    assert Button.A not in adaptive_reading.pressed
    # From that new rest, a firm press (as in the games: deeper than calibrated) presses A
    firm = make_hand(*between(REST, PRESSED, 1.3), noise=0)
    assert Button.A in adaptive.classify(left, firm, 6000).pressed


def test_a_hold_shorter_than_the_window_is_never_cut() -> None:
    adaptive = make_adaptive()
    left, pressed = make_hand(*REST), make_hand(*PRESSED, noise=0)
    for frame in range(120):                        # 6 s resting first
        adaptive.classify(left, make_hand(*REST, noise=0), frame * 50)
    readings = [adaptive.classify(left, pressed, 6000 + frame * 50) for frame in range(60)]   # 3 s held
    assert all(Button.A in reading.pressed for reading in readings)


def test_a_press_held_from_the_very_start_is_not_taken_as_the_rest() -> None:
    """The spike's `a_held` clip: pressed from the first frame. The learned rest waits
    for 4 s of history, so the press lasts at least that long."""
    adaptive = make_adaptive()
    left, pressed = make_hand(*REST), make_hand(*PRESSED, noise=0)
    readings = [adaptive.classify(left, pressed, frame * 50) for frame in range(80)]    # 4 s
    assert all(Button.A in reading.pressed for reading in readings)


def test_after_a_long_hold_the_capped_rest_still_lets_a_deep_press_through() -> None:
    """Known limit: a hold much longer than the window, at the calibrated depth, ends up
    released (the rest climbs to its cap, 0.8). Pressing deeper still counts."""
    adaptive = make_adaptive()
    left, pressed = make_hand(*REST), make_hand(*PRESSED, noise=0)
    for frame in range(200):                        # 10 s held
        adaptive.classify(left, pressed, frame * 50)
    assert adaptive.tracker.offsets()[Button.A] == 0.8
    assert Button.A in adaptive.classify(left, make_hand(*between(REST, PRESSED, 1.3), noise=0), 10000).pressed


def test_only_a_rest_is_relearned() -> None:
    adaptive = make_adaptive()
    left, tilted = make_hand(*REST), make_hand(*between(REST, B_PRESSED, 0.6), noise=0)
    for frame in range(120):
        adaptive.classify(left, tilted, frame * 50)
    assert Button.B not in adaptive.tracker.offsets()
