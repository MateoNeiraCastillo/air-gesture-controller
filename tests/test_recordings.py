from air_controller.recordings import guided_sessions, parse_clip_name


def test_english_clip_names_pass_through() -> None:
    assert parse_clip_name("a_held_night_02") == ("a_held", "night", "02")


def test_spanish_clip_names_are_translated() -> None:
    assert parse_clip_name("boton_a_mantenido_noche_01") == ("a_held", "night", "01")
    assert parse_clip_name("dpad_izq_dia_01") == ("dpad_left", "day", "01")
    assert parse_clip_name("reposo_dia_02") == ("rest", "day", "02")


def test_combo_and_unknown_lighting_are_kept() -> None:
    assert parse_clip_name("combo_office_03") == ("combo", "office", "03")


def test_guided_sessions_mix_old_and_new_names_in_time_order(tmp_path) -> None:
    for name in ("session_20261105_090000.npz", "sesion_20261027_104958.npz", "notes.npz"):
        (tmp_path / name).touch()
    assert [path.name for path in guided_sessions(tmp_path)] == [
        "sesion_20261027_104958.npz", "session_20261105_090000.npz"]
