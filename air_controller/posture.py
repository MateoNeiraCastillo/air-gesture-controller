"""Monitor de postura: avisa cuando la mano ya no está como al calibrar.

En el spike, el `ritmo` de día falló porque el puño giró de frente a la
cámara: la escala de la mano (muñeca → base del medio) cayó de ~60 px a
1–29 px y las medidas del pulgar se dispararon. No lo arreglamos en el
clasificador; lo DETECTAMOS: si el tamaño aparente de una mano se sale del
rango visto durante la calibración (ampliado por los márgenes de la config)
durante `warning_after_ms` seguidos, se avisa para que el jugador se recoloque.

Se compara con un RANGO y no con un único tamaño porque la mano se inclina
algo al mover el pulgar: en el spike de día, la izquierda medía del 77 % al
130 % de su mediana según la postura, todas válidas.
"""

from dataclasses import dataclass, field

import numpy as np

from air_controller.calibration import Calibration
from air_controller.config import PostureConfig
from air_controller.features import hand_size

HAND_NAMES = {"left": "izquierda", "right": "derecha"}


@dataclass
class PostureMonitor:
    calibration: Calibration
    config: PostureConfig
    _out_since_ms: dict[str, int | None] = field(
        default_factory=lambda: {"left": None, "right": None})
    _last_ratio: dict[str, float] = field(
        default_factory=lambda: {"left": float("nan"), "right": float("nan")})

    def update(self, left: np.ndarray | None, right: np.ndarray | None,
               timestamp_ms: int) -> list[str]:
        """Devuelve los avisos activos (lista vacía si todo está bien)."""
        warnings = []
        for hand, landmarks in (("left", left), ("right", right)):
            ratio = self._ratio(hand, landmarks)
            self._last_ratio[hand] = ratio
            low, high = self._limits(hand)
            size = ratio * self._reference(hand)
            out_of_range = not np.isnan(ratio) and not (low <= size <= high)
            if not out_of_range:
                self._out_since_ms[hand] = None
                continue
            if self._out_since_ms[hand] is None:
                self._out_since_ms[hand] = timestamp_ms
            if timestamp_ms - self._out_since_ms[hand] >= self.config.warning_after_ms:
                warnings.append(self._message(hand, ratio))
        return warnings

    def ratio(self, hand: str) -> float:
        """Último tamaño relativo medido (1.0 = como al calibrar)."""
        return self._last_ratio[hand]

    def _reference(self, hand: str) -> float:
        return (self.calibration.left_hand_size if hand == "left"
                else self.calibration.right_hand_size)

    def _limits(self, hand: str) -> tuple[float, float]:
        """Tamaño mínimo y máximo aceptables, en píxeles."""
        smallest, largest = (self.calibration.left_size_range if hand == "left"
                             else self.calibration.right_size_range)
        if largest <= 0:   # calibración sin rango: usar la mediana
            smallest = largest = self._reference(hand)
        return (smallest * self.config.hand_size_ratio_min, largest * self.config.hand_size_ratio_max)

    def _ratio(self, hand: str, landmarks: np.ndarray | None) -> float:
        if landmarks is None or np.isnan(landmarks).any():
            return float("nan")
        return hand_size(landmarks) / self._reference(hand)

    def _message(self, hand: str, ratio: float) -> str:
        name = HAND_NAMES[hand]
        if ratio * self._reference(hand) < self._limits(hand)[0]:
            return (f"Mano {name}: ha girado o se ha alejado ({ratio:.0%} del tamaño calibrado). "
                    "Vuelve a la postura de la calibración.")
        return (f"Mano {name}: demasiado cerca ({ratio:.0%} del tamaño calibrado). "
                "Vuelve a la postura de la calibración.")
