"""
air_keyboard_ui.py

Module 1: KEYBOARD STRUCTURE / RENDERING ONLY.

Responsible for:
- Defining the 3D layout of the keyboard (QWERTY rows placed on a flat
  plane, like keys resting on an invisible table).
- Projecting that 3D layout into 2D so it can be drawn on a video frame,
  tilted toward the camera (a "floating keyboard on a table" look).
- Drawing the keys (fill color, thin border, label) onto a given frame.

This module does NOT touch the webcam and does NOT know about hand
tracking or typing. Another module (air_keyboard_typing.py) imports the
public functions/constants below to actually capture video, detect the
finger, and turn key hits into typed text.

Public API used by other modules:
    FRAME_W, FRAME_H      - expected frame size
    KEYBOARD              - list of key dicts (layout data)
    render_keyboard(frame, t) -> (frame_with_keys, projected_keys)
        projected_keys is a list of (key_dict, poly, depth) tuples:
          - poly: projected 2D quad (numpy int32 array) for x/y hit-testing
          - depth: the key's average distance from the camera in the same
            "world units" used by the 3D layout. Useful for a module that
            wants to do real depth-based press detection (not just 2D
            hover), since keys farther back have a larger depth value.
"""

import math

import cv2
import numpy as np

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
FRAME_W, FRAME_H = 1280, 720

# QWERTY-style layout
ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
]
ROW_OFFSET_X = [0.0, 0.33, 0.8]   # typical physical-keyboard row stagger
KEY_SIZE = 0.6                    # width/depth of a single key (3D units)
KEY_GAP = 0.08                    # gap between keys

# Virtual camera used to project the 3D keyboard -> 2D image.
# A small CAM_DEPTH (keyboard close to the camera) makes the size
# difference between near and far keys strong, which is what sells the
# "3D keyboard lying on a table" look instead of a flat rectangle.
CAM_TILT_DEG = 500   # how much the table is tilted toward the camera (more = flatter/more "top-down")
CAM_DEPTH = 4.0      # distance from camera to the keyboard
FOCAL = 520          # focal length (lower = smaller keyboard)

# "Tech blue" palette
COLOR_KEY_NEAR = (255, 200, 60)    # BGR: bright cyan/blue (near keys)
COLOR_KEY_FAR = (160, 80, 10)      # BGR: dark blue (far keys)
COLOR_BORDER = (40, 25, 10)        # thin dark border (structure only, no glow)
COLOR_TEXT = (255, 255, 255)


# ----------------------------------------------------------------------
# 3D KEYBOARD LAYOUT
# ----------------------------------------------------------------------
def build_keyboard():
    """Build the list of keys with their 3D center position.

    Row depth is assigned so the QWERTY row is the *farthest* row
    (top of a real keyboard) and the spacebar is the *closest* row
    (front of a real keyboard, nearest to the user).
    """
    keys = []
    n_rows = len(ROWS)
    for row_i, row in enumerate(ROWS):
        depth_level = n_rows - row_i
        z = depth_level * (KEY_SIZE + KEY_GAP)
        offset = ROW_OFFSET_X[row_i]
        total_w = len(row) * (KEY_SIZE + KEY_GAP)
        start_x = -total_w / 2 + offset
        for col_i, ch in enumerate(row):
            x = start_x + col_i * (KEY_SIZE + KEY_GAP)
            keys.append({"char": ch, "cx": x, "cz": z, "wide": False})

    keys.append({"char": "SPACE", "cx": 0.0, "cz": 0.0, "wide": True})
    return keys


KEYBOARD = build_keyboard()


def key_corners_3d(key):
    """Return the 4 corners of a key on the Y=0 plane (facing +Y, i.e. 'up')."""
    w = (KEY_SIZE * (1.9 if key["wide"] else 1.0)) / 2
    d = KEY_SIZE / 2
    cx, cz = key["cx"], key["cz"]
    return np.array(
        [
            [cx - w, 0.0, cz - d],
            [cx + w, 0.0, cz - d],
            [cx + w, 0.0, cz + d],
            [cx - w, 0.0, cz + d],
        ],
        dtype=np.float32,
    )


# ----------------------------------------------------------------------
# 3D -> 2D PROJECTION
# ----------------------------------------------------------------------
def _rotate_x(points, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    return points @ R.T


def _project(points_3d, hover_y=0.0):
    """Tilt the flat keyboard plane toward the camera and project it in
    perspective. Rotating the Y=0 plane around the X axis makes the face
    that was pointing 'up' visible as a trapezoid from the camera's point
    of view -- this is what creates the floating/3D table look."""
    pts = points_3d.copy()
    pts[:, 1] += hover_y                 # small floating bob
    pts = _rotate_x(pts, CAM_TILT_DEG)
    pts[:, 2] += CAM_DEPTH               # push the keyboard away from the "eye"
    pts[:, 1] += 1.8                     # vertical centering in frame

    z = np.clip(pts[:, 2], 0.15, None)
    x2d = FRAME_W / 2 + FOCAL * pts[:, 0] / z
    y2d = FRAME_H / 2 + FOCAL * pts[:, 1] / z
    return np.stack([x2d, y2d], axis=1), z


def _color_by_depth(depth):
    near, far = CAM_DEPTH - 1.0, CAM_DEPTH + 3.0
    t = float(np.clip((depth - near) / (far - near), 0.0, 1.0))
    b = COLOR_KEY_NEAR[0] + t * (COLOR_KEY_FAR[0] - COLOR_KEY_NEAR[0])
    g = COLOR_KEY_NEAR[1] + t * (COLOR_KEY_FAR[1] - COLOR_KEY_NEAR[1])
    r = COLOR_KEY_NEAR[2] + t * (COLOR_KEY_FAR[2] - COLOR_KEY_NEAR[2])
    return (int(b), int(g), int(r))


# ----------------------------------------------------------------------
# PUBLIC: RENDER THE KEYBOARD ONTO A FRAME
# ----------------------------------------------------------------------
def render_keyboard(frame, t):
    """Draw the 3D floating keyboard onto `frame`.

    Args:
        frame: BGR image (numpy array) to draw onto, e.g. a webcam frame.
        t: current time in seconds (used for the floating animation).

    Returns:
        (frame_out, projected_keys)
        frame_out: the frame with the keyboard blended on top.
        projected_keys: list of (key_dict, poly, depth) for hit-testing.
            poly is an (4, 2) int32 numpy array with the key's projected
            screen-space quad. depth is the key's average camera-space
            distance (float), for real depth-based press detection.
    """
    hover = math.sin(t * 1.6) * 0.06  # gentle floating motion

    key_layer = frame.copy()
    projected_keys = []

    for key in KEYBOARD:
        corners_3d = key_corners_3d(key)
        corners_2d, depth = _project(corners_3d, hover_y=hover)
        avg_depth = float(np.mean(depth))
        poly = corners_2d.astype(np.int32)

        color = _color_by_depth(avg_depth)
        cv2.fillConvexPoly(key_layer, poly, color)
        cv2.polylines(key_layer, [poly], True, COLOR_BORDER, 1, cv2.LINE_AA)

        center = corners_2d.mean(axis=0)
        scale = np.clip(1.4 - avg_depth / (CAM_DEPTH + 2), 0.4, 1.1)
        label = key["char"] if not key["wide"] else "SPACE"
        text_pos = (
            int(center[0] - 10 * scale * len(label) / 2),
            int(center[1] + 8 * scale),
        )
        # Black outline behind the label so it stays readable on blue
        cv2.putText(
            key_layer, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            key_layer, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * scale, COLOR_TEXT, 2, cv2.LINE_AA,
        )

        projected_keys.append((key, poly, avg_depth))

    # Blend the semi-transparent keys on top of the real camera image
    frame_out = cv2.addWeighted(frame, 0.35, key_layer, 0.65, 0)
    return frame_out, projected_keys