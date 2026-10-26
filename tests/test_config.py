import copy

import pytest

from air_controller.buttons import Button
from air_controller.config import ConfigError, load_config, parse_config

VALID = {
    "calibration": {"frames_per_pose": 90, "min_separation": 3.0},
    "classifier": {"dpad_threshold": 0.5},
    "posture": {"hand_size_ratio_min": 0.7, "hand_size_ratio_max": 1.4, "warning_after_ms": 500},
    "timing": {"press_confirm_ms": 80, "release_confirm_ms": 80, "missing_hand_ms": 200},
    "keys": {"left": "left", "right": "right", "a": "up"},
}


def with_change(section: str, key: str, value) -> dict:
    raw = copy.deepcopy(VALID)
    raw[section][key] = value
    return raw


def test_repository_config_is_valid() -> None:
    config = load_config()
    assert config.keys[Button.A]
    assert config.timing.press_confirm_ms >= 0


def test_valid_config_is_parsed() -> None:
    config = parse_config(VALID)
    assert config.calibration.frames_per_pose == 90
    assert config.keys == {Button.LEFT: "left", Button.RIGHT: "right", Button.A: "up"}


@pytest.mark.parametrize("section,key,value", [
    ("timing", "press_confirm_ms", -5),          # fuera de rango
    ("timing", "press_confirm_ms", "80"),        # texto en vez de número
    ("timing", "press_confirm_ms", True),        # bool no es un número válido
    ("calibration", "frames_per_pose", 90.5),    # debe ser entero
    ("classifier", "dpad_threshold", 1.5),       # fracción imposible
])
def test_invalid_values_are_rejected(section: str, key: str, value) -> None:
    with pytest.raises(ConfigError):
        parse_config(with_change(section, key, value))


def test_typo_in_key_is_rejected() -> None:
    raw = copy.deepcopy(VALID)
    raw["timing"]["press_confim_ms"] = raw["timing"].pop("press_confirm_ms")
    with pytest.raises(ConfigError, match="press_confim_ms"):
        parse_config(raw)


def test_missing_section_is_rejected() -> None:
    raw = copy.deepcopy(VALID)
    del raw["timing"]
    with pytest.raises(ConfigError, match="timing"):
        parse_config(raw)


def test_missing_button_key_is_rejected() -> None:
    raw = copy.deepcopy(VALID)
    del raw["keys"]["a"]
    with pytest.raises(ConfigError, match="a"):
        parse_config(raw)


def test_same_key_for_two_buttons_is_rejected() -> None:
    with pytest.raises(ConfigError, match="misma tecla"):
        parse_config(with_change("keys", "a", "left"))


def test_missing_file_gives_clear_error(tmp_path) -> None:
    with pytest.raises(ConfigError, match="No existe"):
        load_config(tmp_path / "no_existe.toml")
