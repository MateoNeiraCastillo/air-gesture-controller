import numpy as np
import pytest

from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.dashboard import (
    A_CENTER,
    CANVAS_H,
    CANVAS_W,
    GREEN,
    LEFT_BUTTON_BOX,
    RIGHT_BUTTON_BOX,
    CalibrationView,
    GameView,
    dpad_marker,
    fit_window,
    render_calibration,
    render_game,
    wrap_to_width,
)

KEYS = {Button.LEFT: "left", Button.RIGHT: "right", Button.A: "up"}


def reading(pressed=(), left=True, right=True, left_progress=0.0, right_progress=0.0, direction=Button.LEFT):
    return FrameReading(frozenset(pressed), left, right, left_progress, right_progress,
                        direction if left else None)


def view(current: FrameReading, held=()) -> GameView:
    held = frozenset(held)
    return GameView(current, held, held, KEYS, 0.3, {})


def camera() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def is_green(canvas, x, y) -> bool:
    return tuple(int(value) for value in canvas[y, x]) == GREEN


def box_center_edge(box) -> tuple[int, int]:
    """A point inside the button, next to its corner: free of the label."""
    return box[0] + 6, box[1] + 6


@pytest.mark.parametrize("button, lit, unlit", [
    (Button.LEFT, LEFT_BUTTON_BOX, RIGHT_BUTTON_BOX),
    (Button.RIGHT, RIGHT_BUTTON_BOX, LEFT_BUTTON_BOX),
])
def test_each_dpad_button_lights_up_on_its_own(button, lit, unlit):
    canvas = render_game(camera(), view(reading({button}), {button}))
    assert is_green(canvas, *box_center_edge(lit))
    assert not is_green(canvas, *box_center_edge(unlit))
    assert not is_green(canvas, A_CENTER[0] - 25, A_CENTER[1])


def test_a_lights_up_without_touching_the_dpad():
    canvas = render_game(camera(), view(reading({Button.A}), {Button.A}))
    assert is_green(canvas, A_CENTER[0] - 25, A_CENTER[1])
    assert not is_green(canvas, *box_center_edge(LEFT_BUTTON_BOX))
    assert not is_green(canvas, *box_center_edge(RIGHT_BUTTON_BOX))


def test_marker_follows_the_classifier_direction_not_the_largest_progress():
    # Both directions point the same way and RIGHT is a short travel: its
    # progress is larger, but the classifier chose LEFT.
    assert dpad_marker(reading(left_progress=0.5, right_progress=1.5, direction=Button.LEFT)) == -0.5
    assert dpad_marker(reading(left_progress=1.5, right_progress=0.5, direction=Button.RIGHT)) == 0.5


def test_marker_never_points_away_from_the_chosen_direction():
    assert dpad_marker(reading(left_progress=-0.2, direction=Button.LEFT)) == 0.0


def test_no_marker_without_the_left_hand():
    assert dpad_marker(reading(left=False)) is None


@pytest.mark.parametrize("image_shape", [(480, 640, 3), (720, 1280, 3), (1080, 1920, 3)])
def test_canvas_size_does_not_depend_on_the_camera(image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)
    assert render_game(image, view(reading())).shape == (CANVAS_H, CANVAS_W, 3)
    calibration = render_calibration(image, CalibrationView("success", lines=("Done",)))
    assert calibration.shape == (CANVAS_H, CANVAS_W, 3)


@pytest.mark.parametrize("stage", ["position", "prepare", "capture", "success", "rejected"])
def test_all_calibration_stages_render(stage):
    current = CalibrationView(stage, "TITLE", ("A line of text that explains what to do.",),
                              step=2, countdown=3, progress=0.5)
    assert render_calibration(camera(), current).shape == (CANVAS_H, CANVAS_W, 3)


def test_success_is_drawn_in_the_center_of_the_screen():
    canvas = render_calibration(camera(), CalibrationView("success", lines=("Done",)))
    rows, columns = np.where((canvas == GREEN).all(axis=2)[150:630, 240:880])
    assert abs((columns.min() + columns.max()) / 2 + 240 - CANVAS_W / 2) < 30
    assert abs((rows.min() + rows.max()) / 2 + 150 - CANVAS_H / 2) < 60


def test_window_is_centered_and_keeps_the_full_size_on_a_large_screen():
    x, y, width, height = fit_window((0, 33, 1470, 842))
    assert (width, height) == (CANVAS_W, CANVAS_H)
    assert x == (1470 - CANVAS_W) // 2
    assert y >= 33


def test_window_shrinks_to_fit_a_small_screen():
    x, y, width, height = fit_window((0, 25, 1024, 600))
    assert width <= 1024 and height + 30 <= 600
    assert x >= 0 and y >= 25
    assert width / height == pytest.approx(CANVAS_W / CANVAS_H, abs=0.01)


def test_long_text_wraps_inside_the_width():
    lines = wrap_to_width("word " * 60, 0.6, 300)
    assert len(lines) > 1
    assert " ".join(lines) == ("word " * 60).strip()
