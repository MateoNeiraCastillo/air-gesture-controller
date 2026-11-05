"""Names of the recorded files in data/, including the Spanish ones from before the translation.

The spike clips and the first guided sessions were recorded when the project was
in Spanish (`reposo_noche_01.npz`, `sesion_20261027_104958.npz`). data/ is not in
git, so those files can't be renamed for everyone: they are read under their
English names instead, and the analysis scripts keep working on old recordings.
"""

from pathlib import Path

LEGACY_LABELS = {
    "reposo": "rest",
    "boton_a_mantenido": "a_held",
    "boton_a_ritmo": "a_rhythm",
    "dpad_izq": "dpad_left",
    "dpad_der": "dpad_right",
}
LEGACY_LIGHTING = {"dia": "day", "noche": "night"}
GUIDED_PREFIXES = ("session_", "sesion_")


def parse_clip_name(stem: str) -> tuple[str, str, str]:
    """'<label>_<lighting>_<NN>' → (label, lighting, NN), with Spanish names translated."""
    label, lighting, index = stem.rsplit("_", 2)
    return LEGACY_LABELS.get(label, label), LEGACY_LIGHTING.get(lighting, lighting), index


def guided_sessions(directory: Path) -> list[Path]:
    """Guided sessions in `directory`, oldest first (by the timestamp in the name)."""
    paths = [path for prefix in GUIDED_PREFIXES for path in directory.glob(f"{prefix}*.npz")]
    return sorted(paths, key=lambda path: path.stem.split("_", 1)[1])
