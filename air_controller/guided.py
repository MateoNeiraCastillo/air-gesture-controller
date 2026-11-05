"""Sesión guiada: el guion de acciones y cómo puntuar lo que hizo el mando.

El guion se repite en bloques iguales para poder comparar el principio con el
final de la sesión (¿se desvía la postura con el tiempo?).

Puntuación:
- Pasos "mantener": acierto frame a frame del estado CONFIRMADO de cada botón
  (lo que recibiría el juego) frente a lo pedido, ignorando los primeros
  `grace_ms` del paso (tiempo de reacción del jugador).
- Pasos con toques de A: se cuentan las pulsaciones confirmadas de A dentro
  del paso (desplazado por el margen de reacción) frente a las pedidas.
"""

from dataclasses import dataclass

import numpy as np

from air_controller.buttons import Button

GRACE_MS = 700


@dataclass(frozen=True)
class Step:
    text: str
    duration_ms: int
    hold: frozenset[Button] = frozenset()   # botones que deben estar mantenidos
    a_taps: int = 0                          # toques de A pedidos (0 = A según `hold`)


def _step(text: str, seconds: float, *hold: Button, a_taps: int = 0) -> Step:
    return Step(text, int(seconds * 1000), frozenset(hold), a_taps)


BLOCK = (
    _step("Reposo", 4),
    _step("Manten DERECHA", 4, Button.RIGHT),
    _step("Reposo", 3),
    _step("Manten IZQUIERDA", 4, Button.LEFT),
    _step("Reposo", 3),
    _step("Manten A", 3, Button.A),
    _step("Reposo", 3),
    _step("Pulsa A 3 veces", 4, a_taps=3),
    _step("Reposo", 2),
    _step("Manten DERECHA y pulsa A 3 veces", 5, Button.RIGHT, a_taps=3),
    _step("Reposo", 3),
    _step("Manten IZQUIERDA y pulsa A 2 veces", 5, Button.LEFT, a_taps=2),
    _step("Reposo", 2),
    _step("Manten DERECHA y manten A", 4, Button.RIGHT, Button.A),
)


def build_script(blocks: int = 3) -> list[tuple[int, Step]]:
    """Lista de (número de bloque, paso)."""
    return [(block, step) for block in range(1, blocks + 1) for step in BLOCK]


# B no aparece en el guion: se puntúa como "no debe pulsarse nunca" (pulsaciones fantasma).
BUTTON_BITS = {Button.LEFT: 1, Button.RIGHT: 2, Button.A: 4, Button.B: 8}


def to_mask(buttons: frozenset[Button] | set[Button]) -> int:
    return sum(BUTTON_BITS[button] for button in buttons)


def from_mask(mask: int) -> frozenset[Button]:
    return frozenset(button for button, bit in BUTTON_BITS.items() if mask & bit)


@dataclass(frozen=True)
class StepScore:
    block: int
    step: Step
    frames_scored: int
    agreement: dict[Button, float]     # % frames con estado correcto (NaN si no se evalúa)
    a_presses: int                     # pulsaciones de A confirmadas en el paso


def score_step(block: int, step: Step, start_ms: int, timestamps: np.ndarray,
               held_masks: np.ndarray, press_times_a: np.ndarray) -> StepScore:
    """Puntúa un paso a partir de los frames de la sesión.

    `held_masks`: estado confirmado por frame (máscara de bits).
    `press_times_a`: instantes (ms) de las pulsaciones confirmadas de A.
    """
    end_ms = start_ms + step.duration_ms
    in_window = (timestamps >= start_ms + GRACE_MS) & (timestamps < end_ms)
    masks = held_masks[in_window]
    agreement: dict[Button, float] = {}
    for button, bit in BUTTON_BITS.items():
        if button is Button.A and step.a_taps:
            agreement[button] = float("nan")       # los toques se puntúan contando
            continue
        expected = button in step.hold
        actual = (masks & bit) > 0
        agreement[button] = 100 * float(np.mean(actual == expected)) if len(masks) else float("nan")
    shifted = (press_times_a >= start_ms + 300) & (press_times_a < end_ms + 300)
    return StepScore(block, step, int(in_window.sum()), agreement, int(shifted.sum()))
