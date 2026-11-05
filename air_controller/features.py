"""Geometric thumb features, computed on landmarks in pixels.

Coordinates must be in pixels (x*width, y*height), not normalized: MediaPipe
normalizes x and y separately, so in a 640x480 image the angles computed on
normalized coordinates come out distorted.
"""

import numpy as np

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9


def joint_angle_deg(a: np.ndarray, joint: np.ndarray, b: np.ndarray) -> float:
    """Angle at `joint` formed by the segments joint→a and joint→b, in degrees."""
    u = a - joint
    v = b - joint
    norm_product = np.linalg.norm(u) * np.linalg.norm(v)
    if norm_product == 0:
        return float("nan")
    cosine = np.clip(np.dot(u, v) / norm_product, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def thumb_ip_angle(hand: np.ndarray) -> float:
    """Flexion at the thumb IP joint (landmarks 2-3-4). 180° = straight."""
    return joint_angle_deg(hand[THUMB_MCP], hand[THUMB_IP], hand[THUMB_TIP])


def thumb_mcp_angle(hand: np.ndarray) -> float:
    """Flexion at the thumb MCP joint (landmarks 1-2-3). Plan B if the IP doesn't separate."""
    return joint_angle_deg(hand[THUMB_CMC], hand[THUMB_MCP], hand[THUMB_IP])


def hand_size(hand: np.ndarray) -> float:
    """Distance wrist → base of the middle finger: the hand's reference scale."""
    return float(np.linalg.norm(hand[MIDDLE_MCP, :2] - hand[WRIST, :2]))


def thumb_position_features(hand: np.ndarray) -> np.ndarray:
    """Thumb tip position in "hand sizes", invariant to distance.

    Returns (dx, dy) relative to the index finger base and (dx, dy) relative to the wrist.
    This is the feature that worked in the spike: angles fail when the thumb
    bends over the fist, position doesn't.
    """
    size = hand_size(hand)
    if size == 0 or np.isnan(hand).any():
        return np.full(4, np.nan)
    tip = hand[THUMB_TIP, :2]
    from_index = (tip - hand[INDEX_MCP, :2]) / size
    from_wrist = (tip - hand[WRIST, :2]) / size
    return np.concatenate([from_index, from_wrist])


def thumb_offset(hand: np.ndarray) -> np.ndarray:
    """Thumb tip position relative to the index finger base,
    in "hand sizes" (invariant to distance from the camera). Returns (dx, dy)."""
    size = hand_size(hand)
    if size == 0:
        return np.array([np.nan, np.nan])
    return (hand[THUMB_TIP, :2] - hand[INDEX_MCP, :2]) / size
