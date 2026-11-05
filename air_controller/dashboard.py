"""Everything the player sees: the calibration screens and the game dashboard.

The camera image is placed on a larger canvas instead of having panels painted
over it, so nothing covers the hands and nothing depends on the camera's
resolution. All layout is in canvas pixels; the window scales the canvas down
if the screen is too small, so the whole dashboard always fits.

Rendering is pure (image and state in, canvas out), so it can be tested without
a camera or a window. OpenCV can't draw accented characters: they are stripped.
"""

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from air_controller.buttons import Button
from air_controller.calibration import CALIBRATION_SEQUENCE
from air_controller.classifier import FrameReading

WINDOW = "Air Gesture Controller"
CANVAS_W, CANVAS_H = 1120, 780
CENTER_X, CENTER_Y = CANVAS_W // 2, CANVAS_H // 2
TITLE_BAR_H = 30
DEFAULT_SCREEN = (0, 0, 1280, 770)   # used if the screen can't be queried

GREEN = (80, 200, 80)
ORANGE = (40, 140, 255)
RED = (70, 70, 235)
YELLOW = (0, 220, 255)
GRAY = (150, 150, 150)
DIM = (95, 95, 95)
WHITE = (255, 255, 255)
BACKGROUND = (28, 24, 24)
CARD = (48, 42, 42)
BUTTON_IDLE = (72, 66, 66)
HAND_LEFT = ORANGE
HAND_RIGHT = (255, 200, 0)

FONT = cv2.FONT_HERSHEY_DUPLEX
BAR_RANGE = 1.3   # the end of the D-pad bar is 130 % of the calibrated travel

# Game layout: camera on top, then the two hand cards, then the input table.
GAME_CAMERA = (240, 10, 640, 480)
CARD_TOP, CARD_BOTTOM = 500, 648
LEFT_CARD_X = (120, 550)
RIGHT_CARD_X = (570, 1000)
LEFT_BUTTON_BOX = (140, 536, 325, 592)
RIGHT_BUTTON_BOX = (345, 536, 530, 592)
B_CENTER = (705, 580)   # NES order: B on the left, A on the right
A_CENTER = (865, 580)
BUTTON_RADIUS = 38
BAR_X = (140, 530)
BAR_Y = 620
TABLE_X, TABLE_TOP, TABLE_ROW_H = 120, 654, 24
TABLE_COLUMNS = (("BUTTON", 130), ("HAND", 150), ("KEY", 110),
                 ("CAMERA SEES", 170), ("GAME RECEIVES", 190), ("PRESSES", 130))
TABLE_ROWS = ((Button.LEFT, "LEFT", "left"), (Button.RIGHT, "RIGHT", "left"),
              (Button.A, "A", "right"), (Button.B, "B", "right"))

# Calibration layout: the camera sits exactly in the middle of the canvas.
CALIBRATION_CAMERA = (240, 150, 640, 480)


# --------------------------------------------------------------------------- #
# What to show
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GameView:
    reading: FrameReading
    held: frozenset[Button]          # confirmed by the state machine
    sent: frozenset[Button]          # actually reaching the game
    keys: Mapping[Button, str]
    threshold: float
    presses: Mapping[Button, int]
    status: str = "LIVE"
    status_color: tuple[int, int, int] = GREEN
    info: tuple[str, ...] = ()
    hints: tuple[str, ...] = ("R  recalibrate", "Q  quit")
    warnings: tuple[str, ...] = ()
    banner: tuple[str, ...] = ()              # centered over the camera
    banner_progress: float | None = None


@dataclass(frozen=True)
class CalibrationView:
    stage: str                       # "position", "prepare", "capture", "success" or "rejected"
    title: str = ""
    lines: tuple[str, ...] = ()
    step: int = 0                    # 1-based; 0 = before the first pose
    total_steps: int = len(CALIBRATION_SEQUENCE)
    countdown: int | None = None
    progress: float = 0.0
    left_seen: bool = False
    right_seen: bool = False
    left_needed: bool = True
    right_needed: bool = True


def dpad_marker(reading: FrameReading) -> float | None:
    """Marker position on the bar: -1 = calibrated left, 0 = rest, +1 = calibrated right.

    Follows the direction the classifier chose. Comparing the two progress values
    instead would disagree with it when both directions point the same way, and a
    LEFT press would move the marker to the right.
    """
    if not reading.left_visible or reading.dpad_direction is None:
        return None
    if reading.dpad_direction is Button.LEFT:
        return -max(reading.dpad_left_progress, 0.0)
    return max(reading.dpad_right_progress, 0.0)


# --------------------------------------------------------------------------- #
# Window
# --------------------------------------------------------------------------- #

def fit_window(screen: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """(x, y, width, height) that centers the canvas in `screen` = (x, y, width, height)."""
    screen_x, screen_y, screen_w, screen_h = screen
    usable_h = screen_h - TITLE_BAR_H
    scale = min(1.0, screen_w / CANVAS_W, usable_h / CANVAS_H)
    width, height = int(CANVAS_W * scale), int(CANVAS_H * scale)
    return (screen_x + (screen_w - width) // 2,
            screen_y + (usable_h - height) // 2,
            width, height)


def _usable_screen() -> tuple[int, int, int, int]:
    """Screen area without the menu bar and the Dock, with y measured from the top."""
    try:
        from AppKit import NSScreen   # lazy: only exists on macOS (comes with pynput)

        screen = NSScreen.mainScreen()
        full, visible = screen.frame(), screen.visibleFrame()
        top = full.size.height - (visible.origin.y + visible.size.height)
        return (int(visible.origin.x), int(top), int(visible.size.width), int(visible.size.height))
    except Exception:
        return DEFAULT_SCREEN


def open_window() -> None:
    screen = _usable_screen()
    x, y, width, height = fit_window(screen)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, width, height)
    # On macOS, moveWindow measures y from below the menu bar (measured: y=100 puts
    # the window's top at 133 with a 33 px menu bar), so the screen's top is left out.
    cv2.moveWindow(WINDOW, x, y - screen[1])


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #

def to_ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _text(image, text, origin, scale=0.55, color=WHITE, thickness=1) -> None:
    cv2.putText(image, to_ascii(text), origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _centered(image, text, center_x, baseline, scale=0.55, color=WHITE, thickness=1) -> None:
    (width, _), _ = cv2.getTextSize(to_ascii(text), FONT, scale, thickness)
    _text(image, text, (center_x - width // 2, baseline), scale, color, thickness)


def _right_aligned(image, text, right_x, baseline, scale=0.55, color=WHITE, thickness=1) -> None:
    (width, _), _ = cv2.getTextSize(to_ascii(text), FONT, scale, thickness)
    _text(image, text, (right_x - width, baseline), scale, color, thickness)


def wrap_to_width(text: str, scale: float, max_width: int, thickness: int = 1) -> list[str]:
    """Splits `text` into lines no wider than `max_width` pixels."""
    lines, line = [], ""
    for word in to_ascii(text).split():
        candidate = f"{line} {word}".strip()
        if line and cv2.getTextSize(candidate, FONT, scale, thickness)[0][0] > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line] if line else lines


def _shade(image, x0, y0, x1, y1, color=(0, 0, 0), alpha=0.75) -> None:
    region = image[y0:y1, x0:x1]
    cv2.addWeighted(np.full_like(region, color), alpha, region, 1 - alpha, 0, dst=region)


def _new_canvas() -> np.ndarray:
    return np.full((CANVAS_H, CANVAS_W, 3), BACKGROUND, dtype=np.uint8)


def _place_camera(canvas, image, box) -> None:
    """Copies the camera image into `box`, keeping its proportions."""
    x, y, width, height = box
    image_h, image_w = image.shape[:2]
    if (image_w, image_h) != (width, height):
        scale = min(width / image_w, height / image_h)
        image = cv2.resize(image, (int(image_w * scale), int(image_h * scale)),
                           interpolation=cv2.INTER_AREA)
        image_h, image_w = image.shape[:2]
    left, top = x + (width - image_w) // 2, y + (height - image_h) // 2
    canvas[top:top + image_h, left:left + image_w] = image


def _message_card(canvas, lines, center=(CENTER_X, CENTER_Y), width=600, border=GRAY, alpha=0.88) -> None:
    """Centered box of text. `lines`: (text, scale, color, thickness); long ones wrap."""
    rows = []
    for text, scale, color, thickness in lines:
        for row in wrap_to_width(text, scale, width - 60, thickness) or [""]:
            rows.append((row, scale, color, thickness, int(34 * scale) + 12))
    height = sum(row[4] for row in rows) + 44
    x0, y0 = center[0] - width // 2, center[1] - height // 2
    x1, y1 = x0 + width, y0 + height
    _shade(canvas, x0, y0, x1, y1, (20, 18, 18), alpha)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, 2)
    y = y0 + 22
    for row, scale, color, thickness, advance in rows:
        _centered(canvas, row, center[0], y + int(30 * scale), scale, color, thickness)
        y += advance


def draw_hands(image: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> None:
    for hand, color in ((left, HAND_LEFT), (right, HAND_RIGHT)):
        if hand is not None:
            for x, y, _ in hand:
                cv2.circle(image, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Calibration screens
# --------------------------------------------------------------------------- #

def render_calibration(image: np.ndarray, view: CalibrationView) -> np.ndarray:
    canvas = _new_canvas()
    _place_camera(canvas, image, CALIBRATION_CAMERA)
    _draw_calibration_header(canvas, view)
    _draw_hand_checks(canvas, view)
    _draw_capture_bar(canvas, view)
    _centered(canvas, "R  start over     Q  quit", CENTER_X, 760, 0.5, GRAY)

    if view.stage == "position":
        _message_card(canvas, [(view.title, 1.05, YELLOW, 2)] + [(line, 0.62, WHITE, 1) for line in view.lines])
    elif view.stage == "prepare":
        _message_card(canvas, [(view.title, 1.05, YELLOW, 2)]
                      + [(line, 0.62, WHITE, 1) for line in view.lines]
                      + [("Get ready", 0.6, GRAY, 1), (str(view.countdown), 2.0, YELLOW, 3)])
    elif view.stage == "capture":
        _draw_capture_banner(canvas, view)
    elif view.stage == "success":
        _message_card(canvas, [("SUCCESS", 2.4, GREEN, 4)] + [(line, 0.68, WHITE, 1) for line in view.lines],
                      width=640, border=GREEN, alpha=0.93)
    elif view.stage == "rejected":
        _message_card(canvas, [("CALIBRATION FAILED", 1.15, RED, 2)]
                      + [(line, 0.56, WHITE, 1) for line in view.lines]
                      + [("Press R to try again", 0.8, YELLOW, 2)],
                      width=800, border=RED, alpha=0.93)
    return canvas


def _draw_calibration_header(canvas, view: CalibrationView) -> None:
    _centered(canvas, "CALIBRATION", CENTER_X, 55, 1.1, WHITE, 2)
    if view.step == 0:
        _centered(canvas, "Getting ready", CENTER_X, 95, 0.65, GRAY)
    else:
        _centered(canvas, f"Step {view.step} of {view.total_steps}", CENTER_X, 95, 0.65, GRAY)
    spacing = 34
    first = CENTER_X - spacing * (view.total_steps - 1) // 2
    finished = view.stage in ("success", "rejected")
    for index in range(view.total_steps):
        center = (first + index * spacing, 122)
        if finished or index + 1 < view.step:
            cv2.circle(canvas, center, 9, GREEN, -1, cv2.LINE_AA)
        elif index + 1 == view.step:
            cv2.circle(canvas, center, 9, YELLOW, -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, center, 9, DIM, 2, cv2.LINE_AA)


def _draw_hand_checks(canvas, view: CalibrationView) -> None:
    top, bottom = 642, 682
    checks = (("LEFT HAND", view.left_seen, view.left_needed, (240, 550)),
              ("RIGHT HAND", view.right_seen, view.right_needed, (570, 880)))
    for name, seen, needed, (x0, x1) in checks:
        if seen:
            label, color = f"{name}: SEEN", GREEN
        elif needed:
            label, color = f"{name}: NOT SEEN", RED
        else:
            label, color = f"{name}: not needed now", GRAY
        cv2.rectangle(canvas, (x0, top), (x1, bottom), CARD, -1)
        cv2.rectangle(canvas, (x0, top), (x1, bottom), color, 2)
        _centered(canvas, label, (x0 + x1) // 2, top + 27, 0.6, color)


def _draw_capture_bar(canvas, view: CalibrationView) -> None:
    x0, x1, top, bottom = 240, 880, 700, 722
    cv2.rectangle(canvas, (x0, top), (x1, bottom), CARD, -1)
    if view.stage == "capture":
        cv2.rectangle(canvas, (x0, top), (x0 + int((x1 - x0) * min(view.progress, 1.0)), bottom), GREEN, -1)
    cv2.rectangle(canvas, (x0, top), (x1, bottom), GRAY, 1)


def _draw_capture_banner(canvas, view: CalibrationView) -> None:
    """Kept at the top of the camera: the hands must stay visible while capturing."""
    x, y, width, _ = CALIBRATION_CAMERA
    needed_missing = (view.left_needed and not view.left_seen) or (view.right_needed and not view.right_seen)
    _shade(canvas, x, y, x + width, y + 96, (20, 18, 18), 0.8)
    _centered(canvas, view.title, CENTER_X, y + 38, 0.95, YELLOW, 2)
    if needed_missing:
        _centered(canvas, "Paused: bring the missing hand back into view", CENTER_X, y + 76, 0.62, RED)
    else:
        _centered(canvas, f"HOLD IT, don't move   {min(view.progress, 1.0):.0%}", CENTER_X, y + 76, 0.65, WHITE)


# --------------------------------------------------------------------------- #
# Game dashboard
# --------------------------------------------------------------------------- #

def render_game(image: np.ndarray, view: GameView) -> np.ndarray:
    canvas = _new_canvas()
    _place_camera(canvas, image, GAME_CAMERA)
    if view.banner:
        _draw_banner(canvas, view)
    if view.warnings:
        _draw_warnings(canvas, view.warnings)
    _draw_side_panels(canvas, view)
    _draw_left_card(canvas, view)
    _draw_right_card(canvas, view)
    _draw_table(canvas, view)
    return canvas


def _draw_banner(canvas, view: GameView) -> None:
    x, y, width, _ = GAME_CAMERA
    height = 44 + 32 * (len(view.banner) - 1) + (20 if view.banner_progress is not None else 0)
    _shade(canvas, x, y, x + width, y + height, (20, 18, 18), 0.8)
    baseline = y + 32
    for index, line in enumerate(view.banner):
        scale, color = (0.85, WHITE) if index == 0 else (0.55, GRAY)
        _centered(canvas, line, x + width // 2, baseline, scale, color, 2 if index == 0 else 1)
        baseline += 32
    if view.banner_progress is not None:
        bar_y = y + height - 14
        filled = int((width - 40) * min(max(view.banner_progress, 0.0), 1.0))
        cv2.rectangle(canvas, (x + 20, bar_y), (x + 20 + filled, bar_y + 8), GREEN, -1)
        cv2.rectangle(canvas, (x + 20, bar_y), (x + width - 20, bar_y + 8), GRAY, 1)


def _draw_warnings(canvas, warnings) -> None:
    x, y, width, height = GAME_CAMERA
    lines = [line for warning in warnings for line in wrap_to_width(warning, 0.55, width - 40)]
    top = y + height - 26 * len(lines) - 14
    _shade(canvas, x, top, x + width, y + height, (20, 18, 70), 0.85)
    for index, line in enumerate(lines):
        _centered(canvas, line, x + width // 2, top + 26 + 26 * index, 0.55, WHITE)


def _draw_side_panels(canvas, view: GameView) -> None:
    # Left column: state of the controller.
    cv2.rectangle(canvas, (16, 10), (224, 58), view.status_color, -1)
    _centered(canvas, view.status, 120, 44, 0.85, (20, 20, 20), 2)
    baseline = 90
    for line in view.info:
        for row in wrap_to_width(line, 0.5, 200):
            _text(canvas, row, (18, baseline), 0.5, WHITE)
            baseline += 24
        baseline += 8
    # Right column: keys and colour legend.
    _text(canvas, "CONTROLS", (898, 34), 0.55, GRAY)
    baseline = 64
    for hint in view.hints:
        _text(canvas, hint, (898, baseline), 0.55, WHITE)
        baseline += 28
    baseline += 14
    _text(canvas, "COLOURS", (898, baseline), 0.55, GRAY)
    for color, label in ((ORANGE, "camera sees it"), (GREEN, "sent to the game")):
        baseline += 30
        cv2.rectangle(canvas, (898, baseline - 15), (916, baseline + 3), color, -1)
        _text(canvas, label, (924, baseline), 0.48, WHITE)


def _draw_card(canvas, x_range, title, seen) -> None:
    x0, x1 = x_range
    cv2.rectangle(canvas, (x0, CARD_TOP), (x1, CARD_BOTTOM), CARD, -1)
    cv2.rectangle(canvas, (x0, CARD_TOP), (x1, CARD_BOTTOM), GREEN if seen else RED, 2)
    _text(canvas, title, (x0 + 18, CARD_TOP + 25), 0.58, WHITE)
    _right_aligned(canvas, "HAND SEEN" if seen else "HAND NOT SEEN", x1 - 18, CARD_TOP + 25,
                   0.52, GREEN if seen else RED)


def _button_style(button: Button, view: GameView, seen: bool):
    """(fill, border, border thickness, text color) for a button's current state."""
    if not seen:
        return CARD, DIM, 1, DIM
    if button in view.held:
        return GREEN, WHITE, 3, (15, 15, 15)
    if button in view.reading.pressed:
        return BUTTON_IDLE, ORANGE, 4, WHITE
    return BUTTON_IDLE, GRAY, 1, WHITE


def _draw_dpad_button(canvas, box, button: Button, label: str, view: GameView) -> None:
    x0, y0, x1, y1 = box
    fill, border, thickness, ink = _button_style(button, view, view.reading.left_visible)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border, thickness)
    middle_y = (y0 + y1) // 2
    if button is Button.LEFT:
        tip_x, base_x, text_center = x0 + 18, x0 + 42, (x0 + 42 + x1) // 2
    else:
        tip_x, base_x, text_center = x1 - 18, x1 - 42, (x0 + x1 - 42) // 2
    triangle = np.array([[tip_x, middle_y], [base_x, middle_y - 15], [base_x, middle_y + 15]])
    cv2.fillPoly(canvas, [triangle], ink, cv2.LINE_AA)
    _centered(canvas, label, text_center, middle_y + 11, 0.85, ink, 2)


def _draw_dpad_bar(canvas, view: GameView) -> None:
    x0, x1 = BAR_X
    middle = (x0 + x1) // 2
    scale = (x1 - x0) / 2 / BAR_RANGE
    cv2.line(canvas, (x0, BAR_Y), (x1, BAR_Y), GRAY, 2)
    cv2.line(canvas, (middle, BAR_Y - 6), (middle, BAR_Y + 6), GRAY, 1)
    for sign in (-1, 1):
        threshold_x = middle + sign * int(view.threshold * scale)
        cv2.line(canvas, (threshold_x, BAR_Y - 12), (threshold_x, BAR_Y + 12), WHITE, 1)
    marker = dpad_marker(view.reading)
    if marker is None:
        return
    x = middle + int(float(np.clip(marker, -BAR_RANGE, BAR_RANGE)) * scale)
    lit = bool(view.reading.pressed & {Button.LEFT, Button.RIGHT})
    cv2.circle(canvas, (x, BAR_Y), 9, GREEN if lit else ORANGE, -1, cv2.LINE_AA)


def _draw_left_card(canvas, view: GameView) -> None:
    _draw_card(canvas, LEFT_CARD_X, "LEFT THUMB  -  D-PAD", view.reading.left_visible)
    _draw_dpad_button(canvas, LEFT_BUTTON_BOX, Button.LEFT, "LEFT", view)
    _draw_dpad_button(canvas, RIGHT_BUTTON_BOX, Button.RIGHT, "RIGHT", view)
    _draw_dpad_bar(canvas, view)


def _draw_right_card(canvas, view: GameView) -> None:
    seen = view.reading.right_visible
    _draw_card(canvas, RIGHT_CARD_X, "RIGHT THUMB  -  BUTTONS", seen)
    for button, center in ((Button.B, B_CENTER), (Button.A, A_CENTER)):
        fill, border, thickness, ink = _button_style(button, view, seen)
        cv2.circle(canvas, center, BUTTON_RADIUS, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, BUTTON_RADIUS, border, thickness, cv2.LINE_AA)
        _centered(canvas, button.name, center[0], center[1] + 16, 1.4, ink, 3)


def _draw_table(canvas, view: GameView) -> None:
    width = sum(column_width for _, column_width in TABLE_COLUMNS)
    bottom = TABLE_TOP + TABLE_ROW_H * (len(TABLE_ROWS) + 1)
    cv2.rectangle(canvas, (TABLE_X, TABLE_TOP), (TABLE_X + width, bottom), CARD, -1)
    centers, x = [], TABLE_X
    for _, column_width in TABLE_COLUMNS:
        centers.append(x + column_width // 2)
        x += column_width
    for center, (name, _) in zip(centers, TABLE_COLUMNS):
        _centered(canvas, name, center, TABLE_TOP + 17, 0.44, GRAY)
    visible = {"left": view.reading.left_visible, "right": view.reading.right_visible}
    for index, (button, label, hand) in enumerate(TABLE_ROWS):
        top = TABLE_TOP + TABLE_ROW_H * (index + 1)
        if button in view.sent:
            _shade(canvas, TABLE_X, top, TABLE_X + width, top + TABLE_ROW_H, GREEN, 0.28)
        cv2.line(canvas, (TABLE_X, top), (TABLE_X + width, top), DIM, 1)
        cells = _table_cells(button, label, hand, visible[hand], view)
        for center, (text, color) in zip(centers, cells):
            _centered(canvas, text, center, top + 18, 0.5, color)
    cv2.rectangle(canvas, (TABLE_X, TABLE_TOP), (TABLE_X + width, bottom), GRAY, 1)


def _table_cells(button: Button, label: str, hand: str, hand_seen: bool, view: GameView):
    if not hand_seen:
        camera = ("no hand", RED)
    elif button in view.reading.pressed:
        camera = ("PRESSED", ORANGE)
    else:
        camera = ("-", DIM)
    if button in view.sent:
        game = ("KEY DOWN", GREEN)
    elif button in view.held:
        game = ("not sent", YELLOW)
    else:
        game = ("-", DIM)
    return ((label, WHITE),
            (f"{hand}: seen" if hand_seen else f"{hand}: NOT SEEN", GREEN if hand_seen else RED),
            (view.keys.get(button, "-"), WHITE),
            camera,
            game,
            (str(view.presses.get(button, 0)), WHITE))
