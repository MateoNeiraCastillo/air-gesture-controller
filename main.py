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
PRESS_COOLDOWN = 0.35   # seconds before the same key can be "typed" again
COLOR_PRESSED = (120, 255, 120)   # green highlight when a key is hit
COLOR_BORDER = kb.COLOR_BORDER
COLOR_TEXT = kb.COLOR_TEXT
COLOR_HUD_BG = (40, 20, 0)


def find_hovered_key(frame, hand_landmarks, projected_keys):
    """Return (char, fingertip_xy) for the key under the index fingertip,
    or (None, fingertip_xy) if the fingertip isn't over any key."""
    tip = hand_landmarks.landmark[FINGER_TIP_ID]
    fx, fy = int(tip.x * kb.FRAME_W), int(tip.y * kb.FRAME_H)

    for key, poly in projected_keys:
        inside = cv2.pointPolygonTest(poly, (fx, fy), False) >= 0
        if inside:
            cv2.fillConvexPoly(frame, poly, COLOR_PRESSED)
            cv2.polylines(frame, [poly], True, COLOR_BORDER, 1, cv2.LINE_AA)
            return key["char"], (fx, fy)

    return None, (fx, fy)


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
    last_pressed_char = None
    last_press_time = 0.0

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

        hovered_char = None
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hovered_char, (fx, fy) = find_hovered_key(frame, hand_landmarks, projected_keys)
                cv2.circle(frame, (fx, fy), 12, (0, 255, 255), -1, cv2.LINE_AA)

        if hovered_char is not None:
            now = time.time()
            same_key_recent = (
                hovered_char == last_pressed_char and (now - last_press_time) < PRESS_COOLDOWN
            )
            if not same_key_recent:
                typed_text += " " if hovered_char == "SPACE" else hovered_char
                last_pressed_char = hovered_char
                last_press_time = now
        else:
            last_pressed_char = None

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