import math

import numpy as np

from air_controller.features import (
    INDEX_MCP,
    MIDDLE_MCP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    joint_angle_deg,
    thumb_ip_angle,
    thumb_offset,
    thumb_position_features,
)


def blank_hand() -> np.ndarray:
    return np.zeros((21, 3))


def test_straight_angle_is_180() -> None:
    angle = joint_angle_deg(np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([2.0, 0, 0]))
    assert math.isclose(angle, 180.0)


def test_right_angle_is_90() -> None:
    angle = joint_angle_deg(np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([1.0, 1, 0]))
    assert math.isclose(angle, 90.0)


def test_degenerate_angle_is_nan() -> None:
    point = np.array([1.0, 1, 0])
    assert math.isnan(joint_angle_deg(point, point, point))


def test_thumb_ip_angle_bent() -> None:
    hand = blank_hand()
    hand[THUMB_MCP] = [0, 0, 0]
    hand[THUMB_IP] = [10, 0, 0]
    hand[THUMB_TIP] = [10, 10, 0]
    assert math.isclose(thumb_ip_angle(hand), 90.0)


def test_thumb_position_features_scale_invariant_and_nan_safe() -> None:
    hand = blank_hand()
    hand[WRIST] = [0, 100, 0]
    hand[MIDDLE_MCP] = [0, 0, 0]
    hand[INDEX_MCP] = [10, 0, 0]
    hand[THUMB_TIP] = [60, 20, 0]
    features = thumb_position_features(hand)
    assert np.allclose(features, [0.5, 0.2, 0.6, -0.8])
    assert np.allclose(thumb_position_features(hand * 3), features)
    hand[THUMB_TIP] = [np.nan, np.nan, np.nan]
    assert np.isnan(thumb_position_features(hand)).all()


def test_thumb_offset_is_scale_invariant() -> None:
    hand = blank_hand()
    hand[WRIST] = [0, 100, 0]
    hand[MIDDLE_MCP] = [0, 0, 0]
    hand[INDEX_MCP] = [10, 0, 0]
    hand[THUMB_TIP] = [60, 0, 0]
    small = thumb_offset(hand)
    big = thumb_offset(hand * 2)
    assert np.allclose(small, big)
    assert np.allclose(small, [0.5, 0.0])
