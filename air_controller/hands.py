"""Asignación de manos izquierda/derecha por posición en la imagen."""

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
    """Decide qué mano es cuál por la x de la muñeca (imagen en espejo).

    - 2 manos: la de menor x es la izquierda.
    - 1 mano: se asigna según en qué mitad de la imagen esté.
    - 0 o más de 2: ninguna (más de 2 = alguien más en cuadro, no fiable).
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