from dataclasses import dataclass

from air_controller.hands import assign_hands


@dataclass
class Point:
    x: float
    y: float


def make_hand(wrist_x: float) -> list[Point]:
    return [Point(x=wrist_x, y=0.5) for _ in range(21)]


def test_two_hands_are_sorted_by_wrist_x() -> None:
    left_hand = make_hand(0.3)
    right_hand = make_hand(0.7)
    assigned = assign_hands([right_hand, left_hand])
    assert assigned.left is left_hand
    assert assigned.right is right_hand


def test_single_hand_on_left_half_is_left() -> None:
    hand = make_hand(0.3)
    assigned = assign_hands([hand])
    assert assigned.left is hand
    assert assigned.right is None


def test_single_hand_on_right_half_is_right() -> None:
    hand = make_hand(0.7)
    assigned = assign_hands([hand])
    assert assigned.left is None
    assert assigned.right is hand


def test_no_hands() -> None:
    assigned = assign_hands([])
    assert assigned.left is None
    assert assigned.right is None


def test_more_than_two_hands_is_ignored() -> None:
    assigned = assign_hands([make_hand(0.2), make_hand(0.5), make_hand(0.8)])
    assert assigned.left is None
    assert assigned.right is None