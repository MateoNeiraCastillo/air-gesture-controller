"""Los botones del mando invisible."""

from enum import Enum


class Button(Enum):
    """Botones que el controlador sabe detectar.

    El valor es el nombre que se usa en config.toml.
    """

    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
