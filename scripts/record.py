"""Spike paso 1: graba vídeo crudo de la webcam con una etiqueta.

Graba SIN voltear y SIN MediaPipe: el vídeo es la materia prima y se puede
reprocesar con cualquier parámetro. La vista previa sí va en espejo para que
te resulte natural.

Uso:
    python scripts/record.py --label reposo
    python scripts/record.py --label boton_a_mantenido --seconds 30 --light noche
"""

import argparse
import time
from pathlib import Path

import cv2

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COUNTDOWN_SECONDS = 3
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LABELS = ("reposo", "boton_a_mantenido", "boton_a_ritmo", "dpad_izq", "dpad_der", "combo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graba un clip etiquetado para el spike.")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--light", default="noche", help="condición de luz, p. ej. dia / noche")
    return parser.parse_args()


def next_output_path(label: str, light: str) -> Path:
    """data/raw/<label>_<light>_<NN>.mp4 sin pisar grabaciones anteriores."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = 1
    while (path := OUTPUT_DIR / f"{label}_{light}_{index:02d}.mp4").exists():
        index += 1
    return path


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")
    return camera


def show_preview(frame, text: str, color: tuple[int, int, int]) -> bool:
    """Muestra el frame en espejo con un texto. Devuelve True si se pulsó 'q'."""
    preview = cv2.flip(frame, 1)
    cv2.putText(preview, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("record", preview)
    return cv2.waitKey(1) & 0xFF == ord("q")


def countdown(camera: cv2.VideoCapture, label: str) -> bool:
    """Cuenta atrás para colocar las manos (no se graba). False si se canceló."""
    end = time.perf_counter() + COUNTDOWN_SECONDS
    while (remaining := end - time.perf_counter()) > 0:
        ok, frame = camera.read()
        if ok and show_preview(frame, f"{label}: empieza en {remaining:.0f}", (0, 200, 255)):
            return False
    return True


def record(camera: cv2.VideoCapture, path: Path, seconds: float) -> int:
    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("La cámara no devuelve frames.")
    height, width = frame.shape[:2]
    fps = camera.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames_written = 0
    start = time.perf_counter()
    try:
        while (elapsed := time.perf_counter() - start) < seconds:
            ok, frame = camera.read()
            if not ok:
                continue
            writer.write(frame)
            frames_written += 1
            if show_preview(frame, f"REC {elapsed:4.1f}/{seconds:.0f} s", (0, 0, 255)):
                break
    finally:
        writer.release()
    return frames_written


def main() -> None:
    args = parse_args()
    path = next_output_path(args.label, args.light)
    camera = open_camera(CAMERA_INDEX)
    try:
        if not countdown(camera, args.label):
            print("Cancelado, no se guardó nada.")
            return
        frames = record(camera, path, args.seconds)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Guardado {path.name}: {frames} frames")


if __name__ == "__main__":
    main()
