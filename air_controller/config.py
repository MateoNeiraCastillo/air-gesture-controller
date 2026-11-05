"""Configuración del controlador, leída de un archivo TOML.

Todos los umbrales, tiempos y teclas viven en config.toml, no en el código.
Al cargar se valida todo: un valor fuera de rango o una sección mal escrita
falla al arrancar, con un mensaje claro, en vez de comportarse raro en mitad
de una partida.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from air_controller.buttons import Button

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class ConfigError(ValueError):
    """Error en config.toml."""


@dataclass(frozen=True)
class CalibrationConfig:
    frames_per_pose: int        # frames válidos por postura (~3 s a 30 fps)
    min_separation: float       # separación mínima entre posturas, en desviaciones típicas


@dataclass(frozen=True)
class ClassifierConfig:
    dpad_threshold: float       # fracción del recorrido calibrado que cuenta como pulsado


@dataclass(frozen=True)
class PostureConfig:
    hand_size_ratio_min: float  # tamaño de mano / tamaño calibrado por debajo → aviso
    hand_size_ratio_max: float  # por encima → aviso
    warning_after_ms: float     # tiempo fuera de rango antes de avisar (evita avisos por ruido)


@dataclass(frozen=True)
class TimingConfig:
    press_confirm_ms: float     # tiempo pulsado antes de enviar la pulsación
    release_confirm_ms: float   # tiempo soltado antes de enviar la liberación
    missing_hand_ms: float      # tiempo sin ver una mano antes de soltar sus botones


@dataclass(frozen=True)
class Config:
    calibration: CalibrationConfig
    classifier: ClassifierConfig
    posture: PostureConfig
    timing: TimingConfig
    keys: dict[Button, str]
    # Perfiles por juego ([games.<nombre>] en el TOML): el mapa de teclas
    # completo de cada juego, ya fusionado con [keys] y validado.
    games: dict[str, dict[Button, str]] = field(default_factory=dict)


def keys_for_game(config: Config, game: str | None) -> dict[Button, str]:
    """Teclas a usar: las de [keys], o las del perfil del juego si se indica uno."""
    if game is None:
        return config.keys
    if game not in config.games:
        available = ", ".join(sorted(config.games)) or "ninguno"
        raise ConfigError(f"No hay perfil para el juego '{game}'. Disponibles: {available}")
    return config.games[game]


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Lee y valida config.toml. Lanza ConfigError si algo no cuadra."""
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"No existe el archivo de configuración: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path.name} no es TOML válido: {error}") from error
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> Config:
    """Convierte el diccionario del TOML en Config, validando cada valor."""
    _check_keys(raw, {"calibration", "classifier", "posture", "timing", "keys", "games"}, "raíz")
    keys = _parse_keys(_section(raw, "keys"))
    config = Config(
        calibration=_parse_calibration(_section(raw, "calibration")),
        classifier=_parse_classifier(_section(raw, "classifier")),
        posture=_parse_posture(_section(raw, "posture")),
        timing=_parse_timing(_section(raw, "timing")),
        keys=keys,
        games=_parse_games(raw.get("games", {}), keys),
    )
    return config


# --------------------------------------------------------------------------- #
# Secciones
# --------------------------------------------------------------------------- #

def _parse_calibration(section: dict[str, Any]) -> CalibrationConfig:
    _check_keys(section, {"frames_per_pose", "min_separation"}, "calibration")
    return CalibrationConfig(
        frames_per_pose=_int_in_range(section, "frames_per_pose", "calibration", 10, 600),
        min_separation=_number_in_range(section, "min_separation", "calibration", 0.5, 20.0),
    )


def _parse_classifier(section: dict[str, Any]) -> ClassifierConfig:
    _check_keys(section, {"dpad_threshold"}, "classifier")
    return ClassifierConfig(
        dpad_threshold=_number_in_range(section, "dpad_threshold", "classifier", 0.1, 0.9),
    )


def _parse_posture(section: dict[str, Any]) -> PostureConfig:
    _check_keys(section, {"hand_size_ratio_min", "hand_size_ratio_max", "warning_after_ms"}, "posture")
    low = _number_in_range(section, "hand_size_ratio_min", "posture", 0.1, 1.0)
    high = _number_in_range(section, "hand_size_ratio_max", "posture", 1.0, 5.0)
    delay = _number_in_range(section, "warning_after_ms", "posture", 0, 5000)
    return PostureConfig(hand_size_ratio_min=low, hand_size_ratio_max=high, warning_after_ms=delay)


def _parse_timing(section: dict[str, Any]) -> TimingConfig:
    _check_keys(section, {"press_confirm_ms", "release_confirm_ms", "missing_hand_ms"}, "timing")
    return TimingConfig(
        press_confirm_ms=_number_in_range(section, "press_confirm_ms", "timing", 0, 500),
        release_confirm_ms=_number_in_range(section, "release_confirm_ms", "timing", 0, 500),
        missing_hand_ms=_number_in_range(section, "missing_hand_ms", "timing", 0, 2000),
    )


def _parse_keys(section: dict[str, Any]) -> dict[Button, str]:
    expected = {button.value for button in Button}
    _check_keys(section, expected, "keys")
    missing = expected - set(section)
    if missing:
        raise ConfigError(f"[keys]: falta la tecla de {', '.join(sorted(missing))}")
    keys = {button: _key_name(section, button, "keys") for button in Button}
    _check_no_repeated_keys(keys, "keys")
    return keys


def _parse_games(section: Any, base: dict[Button, str]) -> dict[str, dict[Button, str]]:
    """Cada [games.<nombre>] cambia solo las teclas que indica; el resto, las de [keys]."""
    if not isinstance(section, dict):
        raise ConfigError("[games] debe ser una sección")
    games: dict[str, dict[Button, str]] = {}
    for name, overrides in section.items():
        where = f"games.{name}"
        if not isinstance(overrides, dict):
            raise ConfigError(f"[{where}] debe ser una sección")
        _check_keys(overrides, {button.value for button in Button}, where)
        keys = dict(base)
        for button in Button:
            if button.value in overrides:
                keys[button] = _key_name(overrides, button, where)
        _check_no_repeated_keys(keys, where)
        games[name] = keys
    return games


def _key_name(section: dict[str, Any], button: Button, where: str) -> str:
    key = section[button.value]
    if not isinstance(key, str) or not key.strip():
        raise ConfigError(f"[{where}] {button.value}: debe ser un texto no vacío")
    return key.strip().lower()


def _check_no_repeated_keys(keys: dict[Button, str], where: str) -> None:
    values = list(keys.values())
    repeated = {key for key in values if values.count(key) > 1}
    if repeated:
        raise ConfigError(f"[{where}]: la misma tecla para varios botones: {', '.join(sorted(repeated))}")


# --------------------------------------------------------------------------- #
# Utilidades de validación
# --------------------------------------------------------------------------- #

def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in raw:
        raise ConfigError(f"Falta la sección [{name}]")
    section = raw[name]
    if not isinstance(section, dict):
        raise ConfigError(f"[{name}] debe ser una sección, no un valor suelto")
    return section


def _check_keys(section: dict[str, Any], allowed: set[str], where: str) -> None:
    """Un nombre mal escrito (p. ej. 'press_confim_ms') debe fallar, no ignorarse."""
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"[{where}]: claves desconocidas: {', '.join(sorted(unknown))}")


def _number_in_range(section: dict[str, Any], key: str, where: str,
                     low: float, high: float) -> float:
    if key not in section:
        raise ConfigError(f"[{where}]: falta '{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"[{where}] {key}: debe ser un número")
    if not low <= value <= high:
        raise ConfigError(f"[{where}] {key} = {value}: debe estar entre {low} y {high}")
    return float(value)


def _int_in_range(section: dict[str, Any], key: str, where: str, low: int, high: int) -> int:
    value = _number_in_range(section, key, where, low, high)
    if not float(value).is_integer() or isinstance(section[key], float):
        raise ConfigError(f"[{where}] {key}: debe ser un número entero")
    return int(value)
