"""Calibración en directo y vista de lo que lee el clasificador.

1. Te guía por las 4 posturas (reposo, A pulsado, cruceta izquierda,
   cruceta derecha). Antes de cada una tienes 3 s para colocarte.
2. Si la calibración no pasa el control de calidad, te dice por qué
   (en pantalla y en la terminal). Pulsa R para repetir.
3. Si pasa, muestra en directo lo que haría el mando:
   - A: aro naranja = el clasificador lo ve pulsado en este frame;
        relleno verde = pulsación CONFIRMADA (lo que recibiría el juego).
   - Cruceta: una barra que va del reposo (centro) hacia izquierda o derecha.
     La raya blanca marca el umbral. IZQ/DER en verde = confirmado.
   - Avisos del monitor de postura arriba, en rojo.
   En la terminal se imprime cada pulsación y liberación confirmada.

Teclas: R = recalibrar, Q = salir.

Al salir guarda la calibración en data/live/ para poder analizarla.

Uso:
    python scripts/calibrate_live.py
    python scripts/calibrate_live.py --video data/raw/algo.mp4   # sin cámara
"""

import argparse
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from air_controller.buttons import Button
from air_controller.calibration import CalibrationCollector, CalibrationResult, Pose
from air_controller.capture import FrameSource, LatestFrameCamera, VideoFileSource
from air_controller.classifier import Classifier, FrameReading
from air_controller.config import Config, load_config
from air_controller.hands import assign_hands
from air_controller.posture import PostureMonitor
from air_controller.state_machine import ButtonStateMachine

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
LIVE_DIR = ROOT / "data" / "live"
PREPARE_SECONDS = 3.0
WINDOW = "calibracion"

GREEN = (80, 200, 80)
ORANGE = (40, 140, 255)
RED = (60, 60, 230)
GRAY = (150, 150, 150)
WHITE = (255, 255, 255)

INSTRUCTIONS = {
    Pose.REST: ["REPOSO", "Sujeta el mando. Pulgares apoyados, sin pulsar nada."],
    Pose.A_PRESSED: ["A PULSADO", "Pulsa A a fondo con el pulgar derecho.",
                     "El pulgar IZQUIERDO se queda QUIETO, como en reposo."],
    Pose.DPAD_LEFT: ["CRUCETA IZQUIERDA", "Lleva el pulgar izquierdo a la izquierda y mantenlo."],
    Pose.DPAD_RIGHT: ["CRUCETA DERECHA", "Lleva el pulgar izquierdo a la derecha y mantenlo.",
                      "Que sea un movimiento claro, distinto del reposo."],
}


# --------------------------------------------------------------------------- #
# MediaPipe
# --------------------------------------------------------------------------- #

def build_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH),
                                 delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker: vision.HandLandmarker, image: np.ndarray,
                 timestamp_ms: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Landmarks (21, 3) en píxeles de la mano izquierda y derecha, o None."""
    rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(rgb, timestamp_ms)
    assigned = assign_hands(result.hand_landmarks)
    height, width = image.shape[:2]

    def to_pixels(hand) -> np.ndarray | None:
        if hand is None:
            return None
        return np.array([[p.x * width, p.y * height, p.z * width] for p in hand])

    return to_pixels(assigned.left), to_pixels(assigned.right)


# --------------------------------------------------------------------------- #
# Dibujo (OpenCV no dibuja tildes: se quitan solo para la pantalla)
# --------------------------------------------------------------------------- #

def ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def put_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int],
              color: tuple[int, int, int] = WHITE, scale: float = 0.55) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, ascii(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(image, ascii(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        y += int(30 * scale / 0.55)


def wrap(text: str, width: int = 62) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def draw_hands(image: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> None:
    for hand, color in ((left, ORANGE), (right, (255, 200, 0))):
        if hand is not None:
            for x, y, _ in hand:
                cv2.circle(image, (int(x), int(y)), 2, color, -1)


def draw_progress(image: np.ndarray, fraction: float, y: int) -> None:
    width = image.shape[1] - 40
    cv2.rectangle(image, (20, y), (20 + width, y + 14), GRAY, 1)
    cv2.rectangle(image, (20, y), (20 + int(width * min(fraction, 1.0)), y + 14), GREEN, -1)


def draw_reading(image: np.ndarray, reading: FrameReading, threshold: float,
                 held: frozenset[Button]) -> None:
    """Panel inferior: botón A y barra de la cruceta. `held` = confirmados."""
    height, width = image.shape[:2]
    top = height - 90
    cv2.rectangle(image, (0, top), (width, height), (30, 30, 30), -1)

    # Botón A
    center = (width - 60, top + 45)
    if Button.A in held:
        cv2.circle(image, center, 28, GREEN, -1)
    cv2.circle(image, center, 28, ORANGE if Button.A in reading.pressed else GRAY, 3)
    put_lines(image, ["A"], (center[0] - 9, center[1] + 8), WHITE, 0.8)
    if not reading.right_visible:
        put_lines(image, ["sin mano der."], (width - 130, top + 85), RED, 0.45)

    # Cruceta: -1 (izquierda calibrada) ... 0 (reposo) ... +1 (derecha calibrada)
    bar_left, bar_right = 30, width - 140
    mid = (bar_left + bar_right) // 2
    half = (bar_right - bar_left) // 2
    y = top + 45
    scale = half / 1.3   # el extremo de la barra es 130 % del recorrido calibrado
    cv2.line(image, (bar_left, y), (bar_right, y), GRAY, 2)
    for sign in (-1, 1):
        x = mid + sign * int(threshold * scale)          # umbral
        cv2.line(image, (x, y - 14), (x, y + 14), WHITE, 1)
        x = mid + sign * int(scale)                      # postura calibrada
        cv2.line(image, (x, y - 6), (x, y + 6), GRAY, 1)
    if reading.left_visible:
        value = (reading.dpad_right_progress if reading.dpad_right_progress >= reading.dpad_left_progress
                 else -reading.dpad_left_progress)
        x = mid + int(np.clip(value, -1.3, 1.3) * scale)
        lit = Button.LEFT in reading.pressed or Button.RIGHT in reading.pressed
        cv2.circle(image, (int(np.clip(x, bar_left, bar_right)), y), 10, GREEN if lit else ORANGE, -1)
    else:
        put_lines(image, ["sin mano izq."], (bar_left, top + 85), RED, 0.45)
    put_lines(image, ["IZQ"], (bar_left - 5, top + 20), GREEN if Button.LEFT in held else GRAY, 0.5)
    put_lines(image, ["DER"], (bar_right - 30, top + 20), GREEN if Button.RIGHT in held else GRAY, 0.5)
    put_lines(image, ["R = recalibrar   Q = salir"], (bar_left + 120, top + 85), GRAY, 0.45)


# --------------------------------------------------------------------------- #
# Fases
# --------------------------------------------------------------------------- #

class LiveCalibration:
    def __init__(self, source: FrameSource, config: Config, show: bool = True) -> None:
        self.source = source
        self.config = config
        self.show = show
        self.landmarker = build_landmarker()
        self.collector = CalibrationCollector(config.calibration.frames_per_pose)
        self.result: CalibrationResult | None = None
        self.classifier: Classifier | None = None
        self.pose_started_at: float | None = None
        self.current_pose: Pose | None = None
        self.readings: list[FrameReading] = []
        self.machine = ButtonStateMachine(config.timing)
        self.posture: PostureMonitor | None = None
        self.last_timestamp_ms = 0

    def restart(self) -> None:
        self._log_events(self.machine.release_all(self.last_timestamp_ms))
        self.collector = CalibrationCollector(self.config.calibration.frames_per_pose)
        self.result = None
        self.classifier = None
        self.current_pose = None
        print("\n--- Calibración reiniciada ---")

    def step(self) -> bool:
        """Procesa un frame. False cuando hay que salir."""
        frame = self.source.read()
        if frame is None:
            return False
        image = frame.image
        self.last_timestamp_ms = frame.timestamp_ms
        left, right = detect_hands(self.landmarker, image, frame.timestamp_ms)
        draw_hands(image, left, right)

        if self.result is None:
            self._calibrating(image, left, right)
        elif self.result.ok:
            reading = self.classifier.classify(left, right)
            self.readings.append(reading)
            self._log_events(self.machine.update(reading, frame.timestamp_ms))
            draw_reading(image, reading, self.config.classifier.dpad_threshold, self.machine.held)
            warnings = self.posture.update(left, right, frame.timestamp_ms)
            if warnings:
                put_lines(image, [line for w in warnings for line in wrap(w)], (15, 30), RED)
        else:
            lines = ["CALIBRACION RECHAZADA"]
            for problem in self.result.problems:
                lines += wrap(problem)
            lines.append("Pulsa R para repetir.")
            put_lines(image, lines, (15, 30), RED)

        return self._show(image)

    def _calibrating(self, image: np.ndarray, left, right) -> None:
        pose = self.collector.current_pose
        if pose is not self.current_pose:
            self.current_pose = pose
            self.pose_started_at = time.perf_counter()
            print(f"Postura: {pose.value}")
        waiting = PREPARE_SECONDS - (time.perf_counter() - self.pose_started_at)
        lines = INSTRUCTIONS[pose]
        if waiting > 0:
            put_lines(image, lines + [f"Colocate... {waiting:.0f}"], (15, 30), (0, 220, 255))
            return
        counted = self.collector.add(left, right)
        put_lines(image, lines, (15, 30))
        if not counted:
            missing = "izquierda" if left is None else "derecha"
            put_lines(image, [f"No veo la mano {missing}"], (15, image.shape[0] - 40), RED)
        draw_progress(image, self.collector.progress, image.shape[0] - 25)
        if self.collector.done:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        self.result = self.collector.build(self.config.calibration.min_separation,
                                           self.config.classifier.dpad_threshold)
        if self.result.ok:
            self.classifier = Classifier(self.result.calibration, self.config.classifier)
            self.posture = PostureMonitor(self.result.calibration, self.config.posture)
            self.machine = ButtonStateMachine(self.config.timing)
            progress = self.result.calibration.dpad.progress(
                self.result.calibration.left_rest_while_a.center)
            print("Calibración ACEPTADA.")
            print(f"  Pulgar izquierdo al pulsar A: {progress[Button.LEFT]:.0%} hacia izquierda, "
                  f"{progress[Button.RIGHT]:.0%} hacia derecha (límite "
                  f"{self.config.classifier.dpad_threshold / 2:.0%})")
        else:
            print("Calibración RECHAZADA:", *self.result.problems, sep="\n  ")
        self._save_calibration()

    def _save_calibration(self) -> None:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arrays = {f"{pose.name}_{hand}": values
                  for pose, by_hand in self.collector.samples().items()
                  for hand, values in by_hand.items()}
        path = LIVE_DIR / f"calibracion_{stamp}.npz"
        np.savez(path, accepted=np.array(self.result.ok),
                 problems=np.array(self.result.problems, dtype=object), **arrays)
        print(f"  Guardada en {path.relative_to(ROOT)}")

    def _show(self, image: np.ndarray) -> bool:
        if not self.show:
            return True
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("r"):
            self.restart()
        return True

    def _log_events(self, events) -> None:
        for event in events:
            action = "PULSA " if event.pressed else "suelta"
            print(f"  {event.timestamp_ms / 1000:8.2f} s  {action} {event.button.value}")

    def close(self) -> None:
        self._log_events(self.machine.release_all(self.last_timestamp_ms))
        self.source.close()
        self.landmarker.close()
        if self.show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibración en directo.")
    parser.add_argument("--video", type=Path, help="usar un vídeo en vez de la cámara")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        print(f"Falta el modelo en {MODEL_PATH}.")
        return
    config = load_config()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    try:
        while app.step():
            pass
    finally:
        app.close()


if __name__ == "__main__":
    main()
