"""Left/right hand assignment by position in the image."""

from dataclasses import dataclass
from typing import Protocol, Sequence

WRIST = 0
IMAGE_CENTER_X = 0.5


class Landmark(Protocol):
    x: float
    y: float


HandLandmarks = Sequence[Landmark]


@dataclass(frozen=True)
class AssignedHands:
    left: HandLandmarks | None
    right: HandLandmarks | None


def assign_hands(hands: Sequence[HandLandmarks]) -> AssignedHands:
    """Decides which hand is which by the wrist's x (mirrored image).

    - 2 hands: the one with the smaller x is the left.
    - 1 hand: assigned according to which half of the image it is in.
    - 0 or more than 2: none (more than 2 = someone else in frame, unreliable).
    """
    if len(hands) == 2:
        first, second = sorted(hands, key=lambda hand: hand[WRIST].x)
        return AssignedHands(left=first, right=second)
    if len(hands) == 1:
        hand = hands[0]
        if hand[WRIST].x < IMAGE_CENTER_X:
            return AssignedHands(left=hand, right=None)
        return AssignedHands(left=None, right=hand)
    return AssignedHands(left=None, right=None)