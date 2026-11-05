"""Pasa los clips del spike por calibración + clasificador, frame a frame.

Por cada sesión (noche, dia, ...):
  1. Calibra con los primeros frames válidos de reposo, A mantenido,
     cruceta izquierda y cruceta derecha (como haría el juego al arrancar).
  2. Clasifica TODOS los clips de esa sesión, saltándose los frames usados
     para calibrar.
  3. Compara con lo que se pidió en cada clip.

Es la prueba del clasificador sin cámara. Los porcentajes son por frame, en
crudo; las pulsaciones se cuentan además DESPUÉS de la máquina de estados,
que es lo que llegaría al juego. Al final, los avisos del monitor de postura.

Uso:
    python scripts/replay_spike.py
    python scripts/replay_spike.py --ignore-quality   # evaluar aunque la calibración se rechace
"""

import argparse

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from air_controller.buttons import Button
from air_controller.calibration import SEQUENCE_WITHOUT_B, Calibration, CalibrationCollector, Pose
from air_controller.classifier import Classifier, FrameReading
from air_controller.config import load_config
from air_controller.posture import PostureMonitor
from air_controller.state_machine import ButtonStateMachine

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = ROOT / "data" / "landmarks"
SETTLE_FRAMES = 30   # primer segundo de cada clip: todavía colocando las manos
FRAME_MS = 1000 / 30
TARGET = 95.0

CALIBRATION_CLIPS = {
    Pose.REST: "reposo",
    Pose.A_PRESSED: "boton_a_mantenido",
    Pose.DPAD_LEFT: "dpad_izq",
    Pose.DPAD_RIGHT: "dpad_der",
}


@dataclass(frozen=True)
class Clip:
    left: np.ndarray    # (frames, 21, 3), NaN donde no se vio la mano
    right: np.ndarray

    def hands(self, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        left, right = self.left[index], self.right[index]
        return (None if np.isnan(left).any() else left,
                None if np.isnan(right).any() else right)

    def __len__(self) -> int:
        return len(self.left)


def load_sessions() -> dict[str, dict[str, Clip]]:
    """sesión → etiqueta → Clip (si hay varias tomas, la última)."""
    sessions: dict[str, dict[str, Clip]] = defaultdict(dict)
    for path in sorted(LANDMARKS_DIR.glob("*.npz")):
        label, session, _index = path.stem.rsplit("_", 2)
        with np.load(path) as data:
            sessions[session][label] = Clip(data["left"][SETTLE_FRAMES:],
                                            data["right"][SETTLE_FRAMES:])
    return dict(sessions)


def calibrate(clips: dict[str, Clip], frames_per_pose: int, min_separation: float,
              ignore_quality: bool = False
              ) -> tuple[Classifier | None, Calibration | None, dict[str, int]]:
    """Devuelve el clasificador y, por clip, cuántos frames consumió la calibración."""
    config = load_config()
    collector = CalibrationCollector(frames_per_pose, SEQUENCE_WITHOUT_B)   # el spike no grabó B
    used: dict[str, int] = {}
    for pose, label in CALIBRATION_CLIPS.items():
        clip = clips[label]
        index = 0
        while collector.current_pose is pose and index < len(clip):
            collector.add(*clip.hands(index))
            index += 1
        used[label] = index
    result = collector.build(min_separation, config.classifier.dpad_threshold)
    if not result.ok:
        print("  Calibración rechazada:", *result.problems, sep="\n    ")
        if not ignore_quality:
            return None, None, used
        print("  (--ignore-quality: se evalúa igualmente)")
        result = collector.build(min_separation=0.0, dpad_threshold=float("inf"))
    return Classifier(result.calibration, config.classifier), result.calibration, used


def readings(classifier: Classifier, clip: Clip, start: int) -> list[FrameReading]:
    return [classifier.classify(*clip.hands(index)) for index in range(start, len(clip))]


def share(values: list[FrameReading], condition) -> float:
    """% de frames que cumplen `condition`, entre los frames con la mano relevante visible."""
    return 100 * float(np.mean([condition(r) for r in values])) if values else float("nan")


def presses(values: list[FrameReading], button: Button) -> int:
    on = np.array([button in r.pressed for r in values])
    return int(np.sum(on[1:] & ~on[:-1]))


def confirmed(values: list[FrameReading]) -> tuple[dict[Button, int], dict[Button, float]]:
    """Pasa las lecturas por la máquina de estados.

    Devuelve, por botón, cuántas pulsaciones se enviarían al juego y qué % del
    tiempo estaría mantenido.
    """
    machine = ButtonStateMachine(load_config().timing)
    counts = {button: 0 for button in Button}
    held_frames = {button: 0 for button in Button}
    for index, reading in enumerate(values):
        for event in machine.update(reading, int(index * FRAME_MS)):
            if event.pressed:
                counts[event.button] += 1
        for button in machine.held:
            held_frames[button] += 1
    total = max(len(values), 1)
    return counts, {button: 100 * frames / total for button, frames in held_frames.items()}


def posture_warning_share(calibration: Calibration, clip: Clip, start: int) -> float:
    monitor = PostureMonitor(calibration, load_config().posture)
    warned = [bool(monitor.update(*clip.hands(index), int(index * FRAME_MS)))
              for index in range(start, len(clip))]
    return 100 * float(np.mean(warned)) if warned else float("nan")


def report_session(session: str, clips: dict[str, Clip], ignore_quality: bool) -> list[float]:
    config = load_config()
    print(f"\n== Sesión '{session}' ==")
    if any(label not in clips for label in CALIBRATION_CLIPS.values()):
        print("  Faltan clips de calibración.")
        return []
    classifier, calibration, used = calibrate(clips, config.calibration.frames_per_pose,
                                              config.calibration.min_separation, ignore_quality)
    if classifier is None:
        return []

    def run(label: str) -> list[FrameReading]:
        return readings(classifier, clips[label], used.get(label, 0))

    scores: list[float] = []

    def line(text: str, value: float, expected: bool = True) -> None:
        mark = "OK " if value >= TARGET else "NO "
        print(f"  {mark if expected else '   '}{text:<52}{value:6.1f} %")
        if expected:
            scores.append(value)

    right_seen = lambda r: r.right_visible
    left_seen = lambda r: r.left_visible

    rest = run("reposo")
    line("reposo: A sin pulsar", share([r for r in rest if right_seen(r)], lambda r: Button.A not in r.pressed))
    line("reposo: cruceta sin pulsar", share([r for r in rest if left_seen(r)],
                                             lambda r: not ({Button.LEFT, Button.RIGHT} & r.pressed)))
    held = run("boton_a_mantenido")
    line("A mantenido: A pulsado", share([r for r in held if right_seen(r)], lambda r: Button.A in r.pressed))
    for label, button in (("dpad_izq", Button.LEFT), ("dpad_der", Button.RIGHT)):
        values = [r for r in run(label) if left_seen(r)]
        line(f"{label}: solo {button.value} en la cruceta", share(
            values, lambda r, b=button: b in r.pressed and len({Button.LEFT, Button.RIGHT} & r.pressed) == 1))

    for label in ("boton_a_mantenido", "boton_a_ritmo"):
        if label in clips:
            values = [r for r in run(label) if left_seen(r)]
            line(f"{label}: cruceta sin pulsar (solo se usa A)", share(
                values, lambda r: not ({Button.LEFT, Button.RIGHT} & r.pressed)))

    if "combo" in clips:
        combo = run("combo")
        line("combo: cruceta en 'right'", share([r for r in combo if left_seen(r)],
                                                lambda r: Button.RIGHT in r.pressed))

    print("\n  Tras la máquina de estados (lo que llegaría al juego):")
    for label in ("boton_a_ritmo", "combo", "boton_a_mantenido"):
        if label not in clips:
            continue
        values = run(label)
        counts, held_pct = confirmed(values)
        seconds = len(values) / 30
        print(f"     {label:<18} A: {presses(values, Button.A):3d} crudo → {counts[Button.A]:3d} "
              f"confirmadas en {seconds:.0f} s | cruceta mantenida: izq "
              f"{held_pct[Button.LEFT]:5.1f} %, der {held_pct[Button.RIGHT]:5.1f} %")

    print("\n  Monitor de postura (% del tiempo con aviso):")
    for label in sorted(clips):
        print(f"     {label:<18} {posture_warning_share(calibration, clips[label], used.get(label, 0)):5.1f} %")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ignore-quality", action="store_true",
                        help="evaluar aunque la calibración no pase el control de calidad")
    args = parser.parse_args()
    sessions = load_sessions()
    if not sessions:
        print(f"No hay landmarks en {LANDMARKS_DIR}.")
        return
    all_scores = [score for session, clips in sorted(sessions.items())
                  for score in report_session(session, clips, args.ignore_quality)]
    if all_scores:
        print(f"\nPeor caso: {min(all_scores):.1f} %  (objetivo >= {TARGET:.0f} %)")
        print("'crudo' = frame a frame; 'confirmadas' = tras la máquina de estados.")


if __name__ == "__main__":
    main()
