"""Capa 3: qué botones están pulsados en ESTE frame, según la calibración.

Lógica pura y sin memoria: un frame entra, una lectura sale. El suavizado en
el tiempo (no soltar un botón por un frame raro) es cosa de la máquina de
estados (capa 4).

- Botón A: ¿la punta del pulgar derecho está más cerca del "pulsado" o del
  "reposo" calibrados? Distancias medidas en anchuras de ruido.
- Cruceta, en dos pasos:
    1. Dirección: la postura calibrada (izquierda o derecha) más cercana.
    2. Intensidad: cuánto ha avanzado el pulgar desde el reposo hacia ella.
       Cuenta como pulsado a partir de `dpad_threshold` del recorrido.
  Así, empujar más flojo que al calibrar sigue funcionando (en el spike, el
  `combo` de día se empujó menos que `dpad_der`). La dirección NO se elige por
  "mayor avance": en la sesión de noche izquierda y derecha iban en la misma
  dirección (coseno +0,95) y la derecha era un recorrido corto; un punto de
  "izquierda" avanzaba 3 veces más en el eje corto de la derecha.
"""

from dataclasses import dataclass

import numpy as np

from air_controller.buttons import Button
from air_controller.calibration import BUTTON_COLUMNS, DPAD_COLUMNS, Calibration, Reference
from air_controller.config import ClassifierConfig
from air_controller.features import thumb_position_features


@dataclass(frozen=True)
class FrameReading:
    """Resultado de clasificar un frame."""

    pressed: frozenset[Button]
    left_visible: bool
    right_visible: bool
    dpad_left_progress: float    # 0 = reposo, 1 = como en la calibración; NaN si no se ve
    dpad_right_progress: float


@dataclass(frozen=True)
class DpadAxis:
    """Una dirección de la cruceta, precalculada a partir de la calibración."""

    unit: np.ndarray        # dirección reposo → objetivo, en espacio escalado
    length: float           # distancia reposo → objetivo, en el mismo espacio

    def progress(self, offset: np.ndarray) -> float:
        """Fracción del recorrido calibrado que ha hecho `offset` en esta dirección."""
        return float(offset @ self.unit) / self.length


class Classifier:
    def __init__(self, calibration: Calibration, config: ClassifierConfig) -> None:
        self._calibration = calibration
        self._threshold = config.dpad_threshold
        self._button_scale = _pooled_spread(calibration.right_rest, calibration.right_pressed)

    def classify(self, left: np.ndarray | None, right: np.ndarray | None) -> FrameReading:
        """`left`/`right`: landmarks (21, 3) en píxeles, o None si no se ve la mano."""
        pressed: set[Button] = set()
        left_features = _features(left)
        right_features = _features(right)

        if right_features is not None and self._a_pressed(right_features):
            pressed.add(Button.A)

        progress = {Button.LEFT: float("nan"), Button.RIGHT: float("nan")}
        if left_features is not None:
            point = left_features[list(DPAD_COLUMNS)]
            progress = self._calibration.dpad.progress(point)
            nearest = self._calibration.dpad.nearest(point)
            if progress[nearest] >= self._threshold:
                pressed.add(nearest)

        return FrameReading(
            pressed=frozenset(pressed),
            left_visible=left_features is not None,
            right_visible=right_features is not None,
            dpad_left_progress=progress[Button.LEFT],
            dpad_right_progress=progress[Button.RIGHT],
        )

    def _a_pressed(self, features: np.ndarray) -> bool:
        point = features[list(BUTTON_COLUMNS)]
        to_rest = np.linalg.norm((point - self._calibration.right_rest.center) / self._button_scale)
        to_pressed = np.linalg.norm((point - self._calibration.right_pressed.center) / self._button_scale)
        return bool(to_pressed < to_rest)


def _features(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None or np.isnan(landmarks).any():
        return None
    features = thumb_position_features(landmarks)
    return None if np.isnan(features).any() else features


def _pooled_spread(*references: Reference) -> np.ndarray:
    """Escala común: media cuadrática de las dispersiones de las posturas."""
    return np.sqrt(np.mean([reference.spread ** 2 for reference in references], axis=0))
