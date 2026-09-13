"""
air_keyboard_typing.py

Module 2: HAND TRACKING + TYPING LOGIC.

Responsible for:
- Opening the webcam.
- Running MediaPipe Hands to find the index fingertip.
- Checking if the fingertip lands inside one of the keyboard's projected
  key quads (hit-testing), with a small cooldown so a key doesn't repeat
  every single frame while the finger stays on it.
- Building up the typed text and drawing the on-screen HUD.

This module is the "glue": it imports the keyboard STRUCTURE from
air_keyboard_ui.py (module 1) and only adds the interaction/typing layer
on top of it. If module 1's rendering changes, this module keeps working
as long as `render_keyboard()` still returns (frame, projected_keys).

Run directly:
    python3 air_keyboard_typing.py

Requirements (in your virtual environment):
    pip install opencv-python mediapipe numpy
"""

import time

import cv2
import mediapipe as mp

import air_keyboard as kb  # Module 1: keyboard structure / rendering

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FINGER_TIP_ID = 8       # MediaPipe Hands landmark id for the index fingertip
COLOR_PRESSED = (120, 255, 120)    # green highlight while the key is held
COLOR_HOVER = (255, 255, 150)      # faint highlight when just hovering (not pressed)
COLOR_BORDER = kb.COLOR_BORDER
COLOR_TEXT = kb.COLOR_TEXT
COLOR_HUD_BG = (40, 20, 0)

# --- Real, depth-based "press" detection -------------------------------
# MediaPipe gives each landmark a relative z (roughly: negative/smaller
# means closer to the camera). We smooth it over a few frames as a
# "resting depth" baseline, and treat a sudden move toward the camera
# (a push) as a press, with hysteresis so the finger has to pull back
# out before it can press the same key again. This makes typing an
# actual in-air push gesture instead of just hovering over a 2D area.
Z_SMOOTHING = 0.25          # exponential smoothing factor for the baseline
PRESS_Z_DELTA = 0.025       # how much closer (in MediaPipe z units) counts as a push
RELEASE_Z_DELTA = 0.012     # must pull back at least this much to arm the next press


class PressTracker:
    """Tracks one fingertip's depth over time to detect push/release presses."""

    def __init__(self):
        self.baseline_z = None
        self.pressed_key = None

    def update(self, current_z, hovered_char):
        if self.baseline_z is None:
            self.baseline_z = current_z

        pushed_in = (self.baseline_z - current_z) > PRESS_Z_DELTA
        pulled_back = (self.baseline_z - current_z) < RELEASE_Z_DELTA

        # Slowly drift the baseline toward the current depth so that just
        # resting the hand at a new distance doesn't count as a push.
        if pulled_back:
            self.baseline_z += (current_z - self.baseline_z) * Z_SMOOTHING
            self.pressed_key = None

        typed_char = None
        if hovered_char is not None and pushed_in and self.pressed_key != hovered_char:
            typed_char = hovered_char
            self.pressed_key = hovered_char
        elif hovered_char is None:
            self.pressed_key = None

        return typed_char, pushed_in


def get_hovered_key(hand_landmarks, projected_keys):
    """Return (char, poly, fingertip_xy, fingertip_z) for the key under
    the index fingertip, or (None, None, fingertip_xy, fingertip_z) if
    the fingertip isn't over any key. Doesn't draw anything -- drawing
    happens afterward, once we know if it's a hover or an actual press."""
    tip = hand_landmarks.landmark[FINGER_TIP_ID]
    fx, fy = int(tip.x * kb.FRAME_W), int(tip.y * kb.FRAME_H)

    for key, poly, _depth in projected_keys:
        if cv2.pointPolygonTest(poly, (fx, fy), False) >= 0:
            return key["char"], poly, (fx, fy), tip.z

    return None, None, (fx, fy), tip.z


def draw_hud(frame, typed_text):
    cv2.rectangle(frame, (0, 0), (kb.FRAME_W, 55), COLOR_HUD_BG, -1)
    cv2.putText(
        frame, typed_text[-60:], (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "ESC/Q: quit   BACKSPACE: delete", (15, kb.FRAME_H - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, kb.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, kb.FRAME_H)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    typed_text = ""
    tracker = PressTracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (kb.FRAME_W, kb.FRAME_H))

        # --- Module 1 draws the keyboard and gives us the hit-test data ---
        frame, projected_keys = kb.render_keyboard(frame, time.time())

        # --- Module 2 (this file) handles hand tracking + typing ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hovered_char, poly, (fx, fy), tip_z = get_hovered_key(hand_landmarks, projected_keys)

                # Depth-based press: did the finger just push toward the keyboard?
                typed_char, is_pressed = tracker.update(tip_z, hovered_char)

                if poly is not None:
                    color = COLOR_PRESSED if is_pressed else COLOR_HOVER
                    cv2.fillConvexPoly(frame, poly, color)
                    cv2.polylines(frame, [poly], True, COLOR_BORDER, 1, cv2.LINE_AA)

                cv2.circle(frame, (fx, fy), 12, (0, 255, 255), -1, cv2.LINE_AA)

                if typed_char is not None:
                    typed_text += " " if typed_char == "SPACE" else typed_char

        draw_hud(frame, typed_text)

        cv2.imshow("Floating 3D Air Keyboard", frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed in (27, ord("q")):
            break
        if key_pressed in (8, 127):  # backspace
            typed_text = typed_text[:-1]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()