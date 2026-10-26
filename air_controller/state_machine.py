"""Capa 4: de lecturas por frame a pulsaciones de verdad.

El clasificador dice qué parece pulsado en cada frame, con algún parpadeo
suelto. La máquina de estados decide cuándo pulsar y soltar de verdad:

- Pulsar: el botón tiene que verse pulsado durante `press_confirm_ms` seguidos.
- Soltar: tiene que verse soltado durante `release_confirm_ms` seguidos.
- Mano no visible: es "sin información", no "soltado". El botón mantiene su
  estado; si la mano falta más de `missing_hand_ms`, se sueltan sus botones
  para no dejar teclas pegadas.

Trabaja con tiempos en milisegundos, no con frames: con CPU vamos a ~25 fps,
no a 30, y el ritmo de frames puede variar.
"""

from dataclasses import dataclass, field

from air_controller.buttons import Button
from air_controller.classifier import FrameReading
from air_controller.config import TimingConfig

# Qué mano controla cada botón.
HAND_OF = {Button.LEFT: "left", Button.RIGHT: "left", Button.A: "right"}


@dataclass(frozen=True)
class ButtonEvent:
    button: Button
    pressed: bool          # True = pulsar, False = soltar
    timestamp_ms: int


@dataclass
class _ButtonState:
    pressed: bool = False
    change_since_ms: int | None = None    # desde cuándo se ve lo contrario al estado actual


@dataclass
class ButtonStateMachine:
    timing: TimingConfig
    _buttons: dict[Button, _ButtonState] = field(
        default_factory=lambda: {button: _ButtonState() for button in Button})
    _missing_since_ms: dict[str, int | None] = field(
        default_factory=lambda: {"left": None, "right": None})

    @property
    def held(self) -> frozenset[Button]:
        return frozenset(button for button, state in self._buttons.items() if state.pressed)

    def update(self, reading: FrameReading, timestamp_ms: int) -> list[ButtonEvent]:
        """Procesa la lectura de un frame. Devuelve los eventos que hay que enviar."""
        visible = {"left": reading.left_visible, "right": reading.right_visible}
        self._track_missing(visible, timestamp_ms)
        events: list[ButtonEvent] = []
        for button, state in self._buttons.items():
            hand = HAND_OF[button]
            if visible[hand]:
                event = self._observe(button, state, button in reading.pressed, timestamp_ms)
            else:
                event = self._without_hand(button, state, hand, timestamp_ms)
            if event is not None:
                events.append(event)
        return events

    def release_all(self, timestamp_ms: int) -> list[ButtonEvent]:
        """Suelta todo (al salir o al recalibrar): nunca dejar una tecla pegada."""
        events = []
        for button, state in self._buttons.items():
            if state.pressed:
                events.append(ButtonEvent(button, False, timestamp_ms))
            state.pressed = False
            state.change_since_ms = None
        return events

    # ------------------------------------------------------------------ #

    def _observe(self, button: Button, state: _ButtonState, seen_pressed: bool,
                 now: int) -> ButtonEvent | None:
        if seen_pressed == state.pressed:
            state.change_since_ms = None          # coincide con el estado: nada pendiente
            return None
        if state.change_since_ms is None:
            state.change_since_ms = now           # empieza a verse el cambio
        confirm = self.timing.press_confirm_ms if seen_pressed else self.timing.release_confirm_ms
        if now - state.change_since_ms < confirm:
            return None
        state.pressed = seen_pressed
        state.change_since_ms = None
        return ButtonEvent(button, seen_pressed, now)

    def _without_hand(self, button: Button, state: _ButtonState, hand: str,
                      now: int) -> ButtonEvent | None:
        state.change_since_ms = None              # sin información: no confirma nada
        missing_since = self._missing_since_ms[hand]
        if state.pressed and missing_since is not None and \
                now - missing_since >= self.timing.missing_hand_ms:
            state.pressed = False
            return ButtonEvent(button, False, now)
        return None

    def _track_missing(self, visible: dict[str, bool], now: int) -> None:
        for hand, is_visible in visible.items():
            if is_visible:
                self._missing_since_ms[hand] = None
            elif self._missing_since_ms[hand] is None:
                self._missing_since_ms[hand] = now
