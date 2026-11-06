"""Controller configuration, read from a TOML file.

All thresholds, timings and keys live in config.toml, not in the code.
Everything is validated on load: an out-of-range value or a misspelled section
fails at startup with a clear message, instead of misbehaving in the middle
of a game.
"""

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from air_controller.buttons import Button

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


class ConfigError(ValueError):
    """Error in config.toml."""


@dataclass(frozen=True)
class CalibrationConfig:
    frames_per_pose: int        # valid frames per pose (~3 s at 30 fps)
    min_separation: float       # minimum separation between poses, in standard deviations


@dataclass(frozen=True)
class ClassifierConfig:
    dpad_threshold: float       # fraction of the calibrated travel that counts as pressed
    button_threshold: float = 0.5  # A: travel past the current rest that counts as pressed
    b_threshold: float = 0.5       # B: fraction of the calibrated outward tilt
    b_down_fraction: float = 0.0   # B: fraction of the calibrated B's lowering also required
    b_strong_threshold: float = 9.0  # B: a tilt this far counts without the lowering
    rest_window_ms: float = 0   # re-learn the right thumb's rest over this window; 0 = off
    rest_percentile: float = 20  # the rest is this low percentile of the recent progress
    rest_max_shift: float = 0.8  # the rest never moves further than this from the calibration


@dataclass(frozen=True)
class PostureConfig:
    hand_size_ratio_min: float  # hand size / calibrated size below this → warning
    hand_size_ratio_max: float  # above this → warning
    warning_after_ms: float     # time out of range before warning (avoids warnings from noise)


@dataclass(frozen=True)
class TimingConfig:
    press_confirm_ms: float     # time pressed before sending the press
    release_confirm_ms: float   # time released before sending the release
    missing_hand_ms: float      # time without seeing a hand before releasing its buttons


@dataclass(frozen=True)
class Config:
    calibration: CalibrationConfig
    classifier: ClassifierConfig
    posture: PostureConfig
    timing: TimingConfig
    keys: dict[Button, str]
    # Per-game profiles ([games.<name>] in the TOML): the complete key map of
    # each game, already merged with [keys] and validated.
    games: dict[str, dict[Button, str]] = field(default_factory=dict)
    # Per-game rest re-learning window (ms), only for profiles that set rest_window_ms.
    game_rest_window_ms: dict[str, float] = field(default_factory=dict)


def keys_for_game(config: Config, game: str | None) -> dict[Button, str]:
    """Keys to use: those in [keys], or the game's profile if one is given."""
    if game is None:
        return config.keys
    if game not in config.games:
        available = ", ".join(sorted(config.games)) or "none"
        raise ConfigError(f"No profile for game '{game}'. Available: {available}")
    return config.games[game]


def config_for_game(config: Config, game: str | None) -> Config:
    """The config with a game profile's own settings applied (today: rest_window_ms)."""
    keys_for_game(config, game)                       # validates the name
    if game is None or game not in config.game_rest_window_ms:
        return config
    classifier = replace(config.classifier, rest_window_ms=config.game_rest_window_ms[game])
    return replace(config, classifier=classifier)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Reads and validates config.toml. Raises ConfigError if anything is wrong."""
    try:
        with path.open("rb") as file:
            raw = tomllib.load(file)
    except FileNotFoundError as error:
        raise ConfigError(f"Configuration file does not exist: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path.name} is not valid TOML: {error}") from error
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> Config:
    """Converts the TOML dictionary into a Config, validating every value."""
    _check_keys(raw, {"calibration", "classifier", "posture", "timing", "keys", "games"}, "root")
    keys = _parse_keys(_section(raw, "keys"))
    config = Config(
        calibration=_parse_calibration(_section(raw, "calibration")),
        classifier=_parse_classifier(_section(raw, "classifier")),
        posture=_parse_posture(_section(raw, "posture")),
        timing=_parse_timing(_section(raw, "timing")),
        keys=keys,
        games=_parse_games(raw.get("games", {}), keys),
        game_rest_window_ms=_parse_game_rest_windows(raw.get("games", {})),
    )
    return config


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def _parse_calibration(section: dict[str, Any]) -> CalibrationConfig:
    _check_keys(section, {"frames_per_pose", "min_separation"}, "calibration")
    return CalibrationConfig(
        frames_per_pose=_int_in_range(section, "frames_per_pose", "calibration", 10, 600),
        min_separation=_number_in_range(section, "min_separation", "calibration", 0.5, 20.0),
    )


def _parse_classifier(section: dict[str, Any]) -> ClassifierConfig:
    _check_keys(section, {"dpad_threshold", "button_threshold", "b_threshold", "b_down_fraction",
                          "b_strong_threshold", "rest_window_ms", "rest_percentile", "rest_max_shift"},
                "classifier")
    return ClassifierConfig(
        dpad_threshold=_number_in_range(section, "dpad_threshold", "classifier", 0.1, 0.9),
        button_threshold=_number_in_range(section, "button_threshold", "classifier", 0.1, 0.9),
        b_threshold=_number_in_range(section, "b_threshold", "classifier", 0.1, 1.5),
        b_down_fraction=_number_in_range(section, "b_down_fraction", "classifier", 0, 1.0),
        b_strong_threshold=_number_in_range(section, "b_strong_threshold", "classifier", 0.5, 9.0),
        rest_window_ms=_number_in_range(section, "rest_window_ms", "classifier", 0, 60000),
        rest_percentile=_number_in_range(section, "rest_percentile", "classifier", 1, 50),
        rest_max_shift=_number_in_range(section, "rest_max_shift", "classifier", 0, 1.0),
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
        raise ConfigError(f"[keys]: missing key for {', '.join(sorted(missing))}")
    keys = {button: _key_name(section, button, "keys") for button in Button}
    _check_no_repeated_keys(keys, "keys")
    return keys


def _parse_games(section: Any, base: dict[Button, str]) -> dict[str, dict[Button, str]]:
    """Each [games.<name>] changes only the keys it lists; the rest come from [keys]."""
    if not isinstance(section, dict):
        raise ConfigError("[games] must be a section")
    games: dict[str, dict[Button, str]] = {}
    for name, overrides in section.items():
        where = f"games.{name}"
        if not isinstance(overrides, dict):
            raise ConfigError(f"[{where}] must be a section")
        _check_keys(overrides, {button.value for button in Button} | {"rest_window_ms"}, where)
        keys = dict(base)
        for button in Button:
            if button.value in overrides:
                keys[button] = _key_name(overrides, button, where)
        _check_no_repeated_keys(keys, where)
        games[name] = keys
    return games


def _parse_game_rest_windows(section: dict[str, Any]) -> dict[str, float]:
    """[games.<name>] rest_window_ms = 0: e.g. a racing game where you hold the accelerator."""
    return {name: _number_in_range(overrides, "rest_window_ms", f"games.{name}", 0, 60000)
            for name, overrides in section.items()
            if isinstance(overrides, dict) and "rest_window_ms" in overrides}


def _key_name(section: dict[str, Any], button: Button, where: str) -> str:
    key = section[button.value]
    if not isinstance(key, str) or not key.strip():
        raise ConfigError(f"[{where}] {button.value}: must be a non-empty string")
    return key.strip().lower()


def _check_no_repeated_keys(keys: dict[Button, str], where: str) -> None:
    values = list(keys.values())
    repeated = {key for key in values if values.count(key) > 1}
    if repeated:
        raise ConfigError(f"[{where}]: the same key for several buttons: {', '.join(sorted(repeated))}")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in raw:
        raise ConfigError(f"Missing section [{name}]")
    section = raw[name]
    if not isinstance(section, dict):
        raise ConfigError(f"[{name}] must be a section, not a bare value")
    return section


def _check_keys(section: dict[str, Any], allowed: set[str], where: str) -> None:
    """A misspelled name (e.g. 'press_confim_ms') must fail, not be ignored."""
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"[{where}]: unknown keys: {', '.join(sorted(unknown))}")


def _number_in_range(section: dict[str, Any], key: str, where: str,
                     low: float, high: float) -> float:
    if key not in section:
        raise ConfigError(f"[{where}]: missing '{key}'")
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"[{where}] {key}: must be a number")
    if not low <= value <= high:
        raise ConfigError(f"[{where}] {key} = {value}: must be between {low} and {high}")
    return float(value)


def _int_in_range(section: dict[str, Any], key: str, where: str, low: int, high: int) -> int:
    value = _number_in_range(section, key, where, low, high)
    if not float(value).is_integer() or isinstance(section[key], float):
        raise ConfigError(f"[{where}] {key}: must be an integer")
    return int(value)
