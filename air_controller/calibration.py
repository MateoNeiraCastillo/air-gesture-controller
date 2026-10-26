"""Calibración por sesión: aprender cómo son TUS posturas antes de jugar.

En el spike vimos que las referencias de una sesión no sirven en otra (53–67 %
de acierto), así que cada partida empieza calibrando cuatro posturas:

    1. reposo       (las dos manos)  → referencia de "nada pulsado"
    2. A pulsado    (las dos manos)  → la izquierda debe seguir en reposo
    3. cruceta izq. (mano izquierda)
    4. cruceta der. (mano izquierda)

Decisiones:
- Se cuentan frames VÁLIDOS (mano detectada), no segundos: si MediaPipe tarda
  en ver la mano, la calibración espera en vez de quedarse sin datos.
- Referencias con mediana y dispersión robusta (MAD): un landmark disparatado
  ocasional no mueve la referencia.
- Control de calidad: si dos posturas quedan demasiado juntas, la calibración
  se rechaza con un mensaje que dice qué repetir. Incluye el caso que vimos en
  el spike (noche): al pulsar A, el pulgar izquierdo se iba solo a la postura
  de "derecha", y el mando habría pulsado derecha sin querer todo el rato.

Todo es lógica pura: recibe landmarks, no toca la cámara.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from air_controller.buttons import Button
from air_controller.features import hand_size, thumb_position_features

# Columnas de thumb_position_features usadas por cada control (ver spike):
# botón A solo con la altura de la punta; la cruceta con las cuatro.
BUTTON_COLUMNS = (1, 3)
DPAD_COLUMNS = (0, 1, 2, 3)

MAD_TO_STD = 1.4826   # convierte MAD en desviación típica para datos normales
MIN_SPREAD = 1e-3     # suelo de la dispersión: evita dividir entre ~0


class Pose(Enum):
    REST = "reposo"
    A_PRESSED = "A pulsado"
    DPAD_LEFT = "cruceta izquierda"
    DPAD_RIGHT = "cruceta derecha"


CALIBRATION_SEQUENCE = (Pose.REST, Pose.A_PRESSED, Pose.DPAD_LEFT, Pose.DPAD_RIGHT)

# Qué manos tienen que verse en cada postura para que el frame cuente.
REQUIRED_HANDS = {
    Pose.REST: ("left", "right"),
    Pose.A_PRESSED: ("right", "left"),
    Pose.DPAD_LEFT: ("left",),
    Pose.DPAD_RIGHT: ("left",),
}


@dataclass(frozen=True)
class Reference:
    """Una postura resumida: centro (mediana) y dispersión robusta por feature."""

    center: np.ndarray
    spread: np.ndarray

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "Reference":
        center = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - center), axis=0)
        return cls(center=center, spread=np.maximum(MAD_TO_STD * mad, MIN_SPREAD))


@dataclass(frozen=True)
class DpadGeometry:
    """La cruceta calibrada, en un espacio escalado por el ruido de las posturas.

    - nearest(): qué dirección calibrada (izquierda o derecha) está más cerca.
    - progress(): cuánto ha avanzado el pulgar desde el reposo hacia cada una
      (0 = reposo, 1 = como al calibrar).
    """

    scale: np.ndarray
    rest: np.ndarray
    targets: dict[Button, np.ndarray]
    units: dict[Button, np.ndarray]
    lengths: dict[Button, float]

    @classmethod
    def from_references(cls, rest: Reference, left: Reference, right: Reference) -> "DpadGeometry":
        scale = np.sqrt(np.mean([r.spread ** 2 for r in (rest, left, right)], axis=0))
        rest_point = rest.center / scale
        targets = {Button.LEFT: left.center / scale, Button.RIGHT: right.center / scale}
        lengths = {button: float(np.linalg.norm(target - rest_point)) for button, target in targets.items()}
        units = {button: (target - rest_point) / max(lengths[button], MIN_SPREAD)
                 for button, target in targets.items()}
        return cls(scale, rest_point, targets, units, lengths)

    def progress(self, features: np.ndarray) -> dict[Button, float]:
        offset = features / self.scale - self.rest
        return {button: float(offset @ self.units[button]) / max(self.lengths[button], MIN_SPREAD)
                for button in self.targets}

    def nearest(self, features: np.ndarray) -> Button:
        point = features / self.scale
        return min(self.targets, key=lambda button: np.linalg.norm(point - self.targets[button]))


@dataclass(frozen=True)
class Calibration:
    """Todo lo que el clasificador necesita saber de esta sesión."""

    right_rest: Reference         # columnas BUTTON_COLUMNS
    right_pressed: Reference      # columnas BUTTON_COLUMNS
    left_rest: Reference          # columnas DPAD_COLUMNS
    left_rest_while_a: Reference  # pulgar izquierdo mientras la derecha pulsa A
    left_dpad_left: Reference     # columnas DPAD_COLUMNS
    left_dpad_right: Reference    # columnas DPAD_COLUMNS
    dpad: DpadGeometry
    left_hand_size: float         # píxeles; para avisar si la mano gira o se aleja
    right_hand_size: float


@dataclass(frozen=True)
class CalibrationResult:
    calibration: Calibration | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.calibration is not None


def separation(a: Reference, b: Reference) -> float:
    """Distancia entre dos posturas medida en desviaciones típicas combinadas.

    Mira cuántas "anchuras de ruido" hay entre los centros: 1 = se solapan
    mucho, 3 o más = un umbral simple las distingue con fiabilidad.
    """
    pooled = np.sqrt((a.spread ** 2 + b.spread ** 2) / 2)
    return float(np.linalg.norm((a.center - b.center) / pooled))


class CalibrationCollector:
    """Va recogiendo frames postura a postura. Se alimenta frame a frame.

    Uso:
        collector = CalibrationCollector(frames_per_pose=90)
        while not collector.done:
            collector.add(left_landmarks, right_landmarks)   # None si no se ve
            ...mostrar collector.current_pose y collector.progress...
        result = collector.build(min_separation=3.0)
    """

    def __init__(self, frames_per_pose: int) -> None:
        if frames_per_pose < 1:
            raise ValueError("frames_per_pose debe ser >= 1")
        self.frames_per_pose = frames_per_pose
        self._step = 0
        self._features: dict[Pose, dict[str, list[np.ndarray]]] = {
            pose: {"left": [], "right": []} for pose in CALIBRATION_SEQUENCE
        }
        self._hand_sizes: dict[str, list[float]] = {"left": [], "right": []}

    @property
    def done(self) -> bool:
        return self._step >= len(CALIBRATION_SEQUENCE)

    @property
    def current_pose(self) -> Pose | None:
        return None if self.done else CALIBRATION_SEQUENCE[self._step]

    @property
    def progress(self) -> float:
        """Fracción completada de la postura actual (0 a 1)."""
        pose = self.current_pose
        if pose is None:
            return 1.0
        hand = REQUIRED_HANDS[pose][0]
        return len(self._features[pose][hand]) / self.frames_per_pose

    def add(self, left: np.ndarray | None, right: np.ndarray | None) -> bool:
        """Añade un frame. Devuelve True si contó (se veían las manos necesarias)."""
        pose = self.current_pose
        if pose is None:
            return False
        hands = {"left": _valid_features(left), "right": _valid_features(right)}
        if any(hands[name] is None for name in REQUIRED_HANDS[pose]):
            return False
        for name in REQUIRED_HANDS[pose]:
            self._features[pose][name].append(hands[name])
            landmarks = left if name == "left" else right
            self._hand_sizes[name].append(hand_size(landmarks))
        if len(self._features[pose][REQUIRED_HANDS[pose][0]]) >= self.frames_per_pose:
            self._step += 1
        return True

    def build(self, min_separation: float, dpad_threshold: float) -> CalibrationResult:
        if not self.done:
            return CalibrationResult(None, ["La calibración no ha terminado."])
        return build_calibration(self.samples(), self._median_sizes(), min_separation, dpad_threshold)

    def samples(self) -> dict[Pose, dict[str, np.ndarray]]:
        """Features recogidas hasta ahora: postura → mano → array (frames, 4)."""
        return {pose: {hand: np.array(rows) for hand, rows in by_hand.items() if rows}
                for pose, by_hand in self._features.items()}

    def _median_sizes(self) -> dict[str, float]:
        return {hand: float(np.median(sizes)) for hand, sizes in self._hand_sizes.items()}


def build_calibration(samples: dict[Pose, dict[str, np.ndarray]],
                      hand_sizes: dict[str, float],
                      min_separation: float,
                      dpad_threshold: float) -> CalibrationResult:
    """Crea la Calibration y comprueba su calidad.

    `samples[pose][mano]`: array (frames, 4) de thumb_position_features.
    """
    button = list(BUTTON_COLUMNS)
    dpad = list(DPAD_COLUMNS)
    left_rest = Reference.from_samples(samples[Pose.REST]["left"][:, dpad])
    left_dpad_left = Reference.from_samples(samples[Pose.DPAD_LEFT]["left"][:, dpad])
    left_dpad_right = Reference.from_samples(samples[Pose.DPAD_RIGHT]["left"][:, dpad])
    calibration = Calibration(
        right_rest=Reference.from_samples(samples[Pose.REST]["right"][:, button]),
        right_pressed=Reference.from_samples(samples[Pose.A_PRESSED]["right"][:, button]),
        left_rest=left_rest,
        left_rest_while_a=Reference.from_samples(samples[Pose.A_PRESSED]["left"][:, dpad]),
        left_dpad_left=left_dpad_left,
        left_dpad_right=left_dpad_right,
        dpad=DpadGeometry.from_references(left_rest, left_dpad_left, left_dpad_right),
        left_hand_size=hand_sizes["left"],
        right_hand_size=hand_sizes["right"],
    )
    problems = quality_problems(calibration, min_separation, dpad_threshold)
    return CalibrationResult(None if problems else calibration, problems)


def quality_problems(calibration: Calibration, min_separation: float,
                     dpad_threshold: float) -> list[str]:
    """Mensajes para el jugador; lista vacía si la calibración es buena."""
    checks = [
        (calibration.right_rest, calibration.right_pressed,
         "Botón A: pulsado y reposo se parecen demasiado. Repite pulsando más a fondo."),
        (calibration.left_rest, calibration.left_dpad_left,
         "Cruceta izquierda: casi igual que el reposo. Repite llevando el pulgar más a la izquierda."),
        (calibration.left_rest, calibration.left_dpad_right,
         "Cruceta derecha: casi igual que el reposo. Repite llevando el pulgar más a la derecha."),
        (calibration.left_dpad_left, calibration.left_dpad_right,
         "Cruceta: izquierda y derecha se parecen demasiado. Repite exagerando las dos."),
    ]
    problems = [message for a, b, message in checks if separation(a, b) < min_separation]
    problems.extend(_left_thumb_drift_problems(calibration, dpad_threshold))
    if min(calibration.left_hand_size, calibration.right_hand_size) <= 0:
        problems.append("No se pudo medir el tamaño de la mano. Repite con las manos bien visibles.")
    return problems


def _left_thumb_drift_problems(calibration: Calibration, dpad_threshold: float) -> list[str]:
    """¿Se va el pulgar izquierdo, sin querer, hacia una dirección al pulsar A?

    Se exige que la postura "izquierda en reposo mientras pulsas A" se quede por
    debajo de la MITAD del umbral de la cruceta, para dejar margen.
    """
    progress = calibration.dpad.progress(calibration.left_rest_while_a.center)
    names = {Button.LEFT: "izquierda", Button.RIGHT: "derecha"}
    return [
        f"Al pulsar A, tu pulgar izquierdo se mueve hacia '{names[button]}' "
        f"({value:.0%} del recorrido). Repite la calibración marcando más la cruceta "
        f"{names[button]} y dejando el pulgar izquierdo quieto al pulsar A."
        for button, value in progress.items() if value >= dpad_threshold / 2
    ]


def _valid_features(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None or np.isnan(landmarks).any():
        return None
    features = thumb_position_features(landmarks)
    return None if np.isnan(features).any() else features
