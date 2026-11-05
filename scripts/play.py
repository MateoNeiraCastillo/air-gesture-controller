"""El controlador completo: juega con el mando invisible.

Flujo:
  1. Calibración (como en calibrate_live.py; R repite si se rechaza).
  2. Juego: cámara → manos → clasificador → máquina de estados → teclas.
  3. Al salir imprime la latencia por etapas (p50 / p95).

Teclas (con la ventana de la cámara enfocada):
  P = pausa / reanudar (suelta todas las teclas)
  R = recalibrar       (suelta todas las teclas)
  Q = salir
Freno de emergencia: aparta las manos de la cámara. Pasados `missing_hand_ms`
(200 ms) se sueltan todas las teclas.

IMPORTANTE: las teclas van a la ventana que tenga el foco. Tras calibrar,
haz clic en la ventana del juego. En macOS, la app desde la que ejecutas
(Terminal o VS Code) necesita permiso de Accesibilidad.

Uso:
    python scripts/play.py --dry-run     # no envía teclas: las imprime
    python scripts/play.py               # juega de verdad
    python scripts/play.py --game celeste  # teclas del perfil [games.celeste]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_live import (  # noqa: E402  (script hermano, mismo directorio)
    GRAY,
    GREEN,
    RED,
    WINDOW,
    LiveCalibration,
    detect_hands,
    draw_hands,
    draw_reading,
    put_lines,
    wrap,
)

from air_controller.capture import Frame, LatestFrameCamera, VideoFileSource  # noqa: E402
from air_controller.config import keys_for_game, load_config  # noqa: E402
from air_controller.latency import LatencyTracker  # noqa: E402
from air_controller.output import KeyOutput, PynputKeySender, RecordingKeySender  # noqa: E402

YELLOW = (0, 220, 255)


class Player:
    def __init__(self, app: LiveCalibration, output: KeyOutput, show: bool = True) -> None:
        self.app = app
        self.output = output
        self.show = show
        self.paused = False
        self.latency = LatencyTracker()

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Calibra y juega hasta Q (o hasta que se acabe el vídeo)."""
        while True:
            if not self._calibrate():
                return
            action = self._play()
            if action == "quit":
                return
            self.app.restart()                 # "recalibrate"

    def _calibrate(self) -> bool:
        while not (self.app.result is not None and self.app.result.ok):
            if not self.app.step():
                return False
        print("\nCalibración aceptada. ¡A jugar! Haz clic en la ventana del juego.")
        return True

    def _play(self) -> str:
        """Bucle de juego. Devuelve 'quit' o 'recalibrate'."""
        while True:
            frame = self.app.source.read()
            if frame is None:
                return "quit"
            self.app.last_timestamp_ms = frame.timestamp_ms
            image = self._process(frame)
            action = self._handle_keys(image)
            if action is not None:
                self._release_everything()
                return action

    def _process(self, frame: Frame):
        start = time.perf_counter()
        self.latency.add_frame(start)
        self.latency.add_stage("espera del frame", (start - frame.captured_s) * 1000)

        left, right = detect_hands(self.app.landmarker, frame.image, frame.timestamp_ms)
        detected = time.perf_counter()
        self.latency.add_stage("detección (MediaPipe)", (detected - start) * 1000)

        reading = self.app.classifier.classify(left, right)
        events = self.app.machine.update(reading, frame.timestamp_ms)
        warnings = self.app.posture.update(left, right, frame.timestamp_ms)
        decided = time.perf_counter()
        self.latency.add_stage("clasificar + estados", (decided - detected) * 1000)

        if not self.paused and events:
            self.output.apply(events)
            sent = time.perf_counter()
            self.latency.add_stage("envío de teclas", (sent - decided) * 1000)
            self.latency.add_end_to_end((sent - frame.captured_s) * 1000)

        image = frame.image
        draw_hands(image, left, right)
        draw_reading(image, reading, self.app.config.classifier.dpad_threshold, self.app.machine.held)
        status = ["PAUSA (P para seguir)"] if self.paused else ["JUGANDO"]
        status.append(f"{self.latency.fps():.0f} fps   teclas: "
                      f"{' '.join(sorted(self.output.pressed_keys)) or '-'}")
        put_lines(image, status, (15, 30), YELLOW if self.paused else GREEN)
        if warnings:
            put_lines(image, [line for w in warnings for line in wrap(w)], (15, 90), RED, 0.5)
        put_lines(image, ["P pausa  R recalibrar  Q salir"], (15, image.shape[0] - 100), GRAY, 0.45)
        return image

    def _handle_keys(self, image) -> str | None:
        if not self.show:
            return None
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return "quit"
        if key == ord("r"):
            return "recalibrate"
        if key == ord("p"):
            self.paused = not self.paused
            if self.paused:
                self._release_everything()
            print("Pausa" if self.paused else "Reanudado")
        return None

    def _release_everything(self) -> None:
        """Suelta en la máquina de estados Y en el teclado: nada queda pulsado."""
        self.app.machine.release_all(self.app.last_timestamp_ms)
        self.output.release_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Air Gesture Controller.")
    parser.add_argument("--dry-run", action="store_true",
                        help="no enviar teclas: imprimirlas en la terminal")
    parser.add_argument("--video", type=Path, help="usar un vídeo en vez de la cámara (pruebas)")
    parser.add_argument("--game", help="perfil de teclas de config.toml ([games.<nombre>]), p. ej. celeste")
    args = parser.parse_args()

    config = load_config()
    keys = keys_for_game(config, args.game)
    print("Teclas: " + ", ".join(f"{button.value} → {key}" for button, key in keys.items()))
    sender = RecordingKeySender(echo=True) if args.dry_run else PynputKeySender()
    source = VideoFileSource(args.video) if args.video else LatestFrameCamera()
    app = LiveCalibration(source, config)
    player = None
    try:
        with KeyOutput(sender, keys) as output:
            player = Player(app, output)
            player.run()
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        app.close()
        if player is not None:
            print(player.latency.summary())


if __name__ == "__main__":
    main()
