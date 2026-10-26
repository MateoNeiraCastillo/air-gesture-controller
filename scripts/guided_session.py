"""Sesión guiada de ~2,5 minutos: el examen final de la Fase 2.

1. Calibras como en calibrate_live.py (R para repetir si se rechaza).
2. Durante ~2,5 min la pantalla te pide acciones (mantener derecha, pulsar A
   3 veces, derecha + A...). Arriba ves la actual y debajo la siguiente.
   El panel inferior muestra lo que recibiría el juego, como en la vista
   en directo.
3. Al terminar (o con Q) se guarda todo en data/guided/ para analizarlo con:
       python scripts/evaluate_guided.py

Uso:
    python scripts/guided_session.py
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_live import (  # noqa: E402  (script hermano, mismo directorio)
    GRAY,
    RED,
    WHITE,
    LiveCalibration,
    detect_hands,
    draw_hands,
    draw_progress,
    draw_reading,
    put_lines,
    wrap,
)

from air_controller.capture import LatestFrameCamera, VideoFileSource  # noqa: E402
from air_controller.config import load_config  # noqa: E402
from air_controller.guided import build_script, to_mask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "guided"
WINDOW = "calibracion"
LEAD_IN_MS = 3000
YELLOW = (0, 220, 255)


class GuidedSession:
    def __init__(self, app: LiveCalibration, blocks: int) -> None:
        self.app = app
        self.script = build_script(blocks)
        self.rows: dict[str, list] = {key: [] for key in (
            "timestamp_ms", "step", "raw_mask", "held_mask", "left_visible", "right_visible",
            "dpad_left_progress", "dpad_right_progress", "posture_warning", "left", "right")}
        self.events: list[tuple[int, str, bool]] = []
        self.offsets_ms = np.cumsum([0] + [step.duration_ms for _, step in self.script])
        self.last_index = -1
        self.start_ms: int | None = None

    def run(self) -> bool:
        """True si la sesión llegó al final; False si se abortó con Q."""
        empty = np.full((21, 3), np.nan)
        while True:
            frame = self.app.source.read()
            if frame is None:
                return False
            now = frame.timestamp_ms
            if self.start_ms is None:
                self.start_ms = now + LEAD_IN_MS
            elapsed = now - self.start_ms
            index = self._step_index(elapsed)
            if index is not None and index >= len(self.script):
                return True

            image = frame.image
            left, right = detect_hands(self.app.landmarker, image, now)
            draw_hands(image, left, right)
            reading = self.app.classifier.classify(left, right)
            events = self.app.machine.update(reading, now)
            for event in events:
                self.events.append((event.timestamp_ms, event.button.value, event.pressed))
            warnings = self.app.posture.update(left, right, now)

            if index is not None:
                self._record(now, index, reading, warnings, left, right, empty)
            self._draw(image, elapsed, index, reading, warnings)
            if self.app.show:
                cv2.imshow(WINDOW, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return False

    def _step_index(self, elapsed: int) -> int | None:
        """Índice del paso en curso (len(script) al terminar); None en la cuenta atrás."""
        if elapsed < 0:
            return None
        index = int(np.searchsorted(self.offsets_ms, elapsed, side="right")) - 1
        self.last_index = max(self.last_index, min(index, len(self.script) - 1))
        return index

    def _record(self, now, index, reading, warnings, left, right, empty) -> None:
        held = self.app.machine.held
        self.rows["timestamp_ms"].append(now)
        self.rows["step"].append(index)
        self.rows["raw_mask"].append(to_mask(reading.pressed))
        self.rows["held_mask"].append(to_mask(held))
        self.rows["left_visible"].append(reading.left_visible)
        self.rows["right_visible"].append(reading.right_visible)
        self.rows["dpad_left_progress"].append(reading.dpad_left_progress)
        self.rows["dpad_right_progress"].append(reading.dpad_right_progress)
        self.rows["posture_warning"].append(bool(warnings))
        self.rows["left"].append(empty if left is None else left)
        self.rows["right"].append(empty if right is None else right)

    def _draw(self, image, elapsed, index, reading, warnings) -> None:
        draw_reading(image, reading, self.app.config.classifier.dpad_threshold, self.app.machine.held)
        if index is None:
            first = self.script[0][1].text
            put_lines(image, [f"Empieza en {-elapsed / 1000:.0f}...", f"Primero: {first}"],
                      (15, 35), YELLOW, 0.8)
            return
        block, step = self.script[index]
        progress = (elapsed - self.offsets_ms[index]) / step.duration_ms
        put_lines(image, [step.text], (15, 40), WHITE, 0.9)
        if index + 1 < len(self.script):
            put_lines(image, [f"Siguiente: {self.script[index + 1][1].text}"], (15, 75), GRAY, 0.55)
        put_lines(image, [f"Bloque {block}/3   paso {index + 1}/{len(self.script)}"],
                  (15, 100), GRAY, 0.45)
        draw_progress(image, progress, 110)
        if warnings:
            put_lines(image, [line for w in warnings for line in wrap(w)], (15, 150), RED, 0.5)

    def save(self, completed: bool) -> Path:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"sesion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
        rows = {key: np.array(values) for key, values in self.rows.items()}
        np.savez_compressed(
            path,
            completed=np.array(completed),
            blocks=np.array(max((b for b, _ in self.script), default=0)),
            step_starts_ms=np.array([self.start_ms + offset
                                     for offset in self.offsets_ms[:self.last_index + 1]]),
            event_times_ms=np.array([t for t, _, _ in self.events]),
            event_buttons=np.array([b for _, b, _ in self.events]),
            event_pressed=np.array([p for _, _, p in self.events]),
            press_confirm_ms=np.array(self.app.config.timing.press_confirm_ms),
            dpad_threshold=np.array(self.app.config.classifier.dpad_threshold),
            **rows,
        )
        return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sesión guiada de la Fase 2.")
    parser.add_argument("--video", type=Path, help="usar un vídeo en vez de la cámara (pruebas)")
    parser.add_argument("--blocks", type=int, default=3)
    args = parser.parse_args()
    config = load_config()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    session = None
    try:
        # Fase 1: calibrar hasta que se acepte (R repite, Q sale).
        while not (app.result is not None and app.result.ok):
            if not app.step():
                print("Calibración interrumpida; no hay sesión.")
                return
        print("\nCalibración aceptada. Empieza la sesión guiada.")
        session = GuidedSession(app, args.blocks)
        completed = session.run()
        app.machine.release_all(app.last_timestamp_ms)
        path = session.save(completed)
        status = "completa" if completed else "interrumpida (se guarda igualmente)"
        print(f"Sesión {status}: {path.relative_to(ROOT)}")
        print("Analízala con: python scripts/evaluate_guided.py")
    finally:
        app.close()


if __name__ == "__main__":
    main()
