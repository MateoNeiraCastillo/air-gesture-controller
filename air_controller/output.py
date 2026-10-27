"""Capa 5: convertir eventos de botón en teclas pulsadas y soltadas.

Regla de oro: NUNCA dejar una tecla pegada. Una tecla pegada en un juego es
peor que una pulsación perdida (Mario corre solo hacia un precipicio). Por eso:
- Cada emisor recuerda qué teclas tiene pulsadas y `release_all()` las suelta.
- `KeyOutput` es un context manager: al salir del `with` (normal, por
  excepción o con Ctrl+C) suelta todo.
- Además, la máquina de estados suelta los botones de una mano que deja de
  verse > `missing_hand_ms`: apartar las manos de la cámara es un "freno de
  emergencia" natural.

El emisor real usa pynput, que se importa solo al crearlo: los tests usan
`RecordingKeySender` y no necesitan pynput ni permisos del sistema.
"""

from dataclasses import dataclass, field
from typing import Protocol

from air_controller.buttons import Button
from air_controller.state_machine import ButtonEvent

SPECIAL_KEYS = {"left", "right", "up", "down", "space", "enter", "shift", "ctrl", "alt",
                "tab", "esc"}


class KeySender(Protocol):
    def press(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...


def validate_key_name(key: str) -> None:
    """Acepta teclas especiales (left, space...) o un solo carácter (z, x...)."""
    if key not in SPECIAL_KEYS and len(key) != 1:
        raise ValueError(f"Tecla desconocida: '{key}'. Usa una de {sorted(SPECIAL_KEYS)} "
                         "o un solo carácter.")


class PynputKeySender:
    """Envía teclas reales al sistema.

    En macOS hace falta dar permiso de Accesibilidad a la app desde la que se
    ejecuta (Terminal o VS Code); sin él, las teclas se ignoran sin error.
    """

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key, KeyCode  # import perezoso

        self._controller = Controller()
        self._special = {name: getattr(Key, name) for name in SPECIAL_KEYS}
        self._key_code = KeyCode

    def _resolve(self, key: str):
        return self._special.get(key) or self._key_code.from_char(key)

    def press(self, key: str) -> None:
        self._controller.press(self._resolve(key))

    def release(self, key: str) -> None:
        self._controller.release(self._resolve(key))


@dataclass
class RecordingKeySender:
    """Emisor falso para tests y para --dry-run: apunta lo que se habría enviado."""

    log: list[tuple[str, str]] = field(default_factory=list)   # (accion, tecla)
    echo: bool = False

    def press(self, key: str) -> None:
        self.log.append(("press", key))
        if self.echo:
            print(f"  [tecla] ↓ {key}")

    def release(self, key: str) -> None:
        self.log.append(("release", key))
        if self.echo:
            print(f"  [tecla] ↑ {key}")


class KeyOutput:
    """Traduce ButtonEvent → teclas, recordando lo pulsado para poder soltarlo todo."""

    def __init__(self, sender: KeySender, keys: dict[Button, str]) -> None:
        for key in keys.values():
            validate_key_name(key)
        self._sender = sender
        self._keys = keys
        self._down: set[str] = set()

    def apply(self, events: list[ButtonEvent]) -> None:
        for event in events:
            key = self._keys[event.button]
            if event.pressed and key not in self._down:
                self._sender.press(key)
                self._down.add(key)
            elif not event.pressed and key in self._down:
                self._sender.release(key)
                self._down.discard(key)

    def release_all(self) -> None:
        for key in sorted(self._down):
            self._sender.release(key)
        self._down.clear()

    @property
    def pressed_keys(self) -> frozenset[str]:
        return frozenset(self._down)

    def __enter__(self) -> "KeyOutput":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release_all()
