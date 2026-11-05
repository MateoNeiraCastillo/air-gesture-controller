"""The buttons of the invisible controller."""

from enum import Enum


class Button(Enum):
    """Buttons the controller knows how to detect.

    The value is the name used in config.toml.
    """

    LEFT = "left"
    RIGHT = "right"
    A = "a"
