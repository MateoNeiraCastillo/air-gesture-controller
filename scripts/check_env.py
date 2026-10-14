"""Fase 0: verifica entorno, cámara y MediaPipe HandLandmarker con dos manos.

Abre la webcam, detecta manos en vivo y al cerrar imprime un resumen:
FPS real, coste de inferencia (p50/p95), % de frames con dos manos y si la
lateralidad (izquierda/derecha) sale coherente.

Uso:
    python scripts/check_env.py                 # cámara 0, 20 s
    python scripts/check_env.py --seconds 30
    python scripts/check_env.py --no-mirror     # para comparar la lateralidad

Pulsa 'q' en la ventana de vídeo para terminar antes.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WRIST = 0  # índice del landmark de la muñeca


@dataclass
class SessionStats:
    """Contadores acumulados durante la prueba en vivo."""

    frames: int = 0
    dropped_frames: int = 0
    hands_count: dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0, 2: 0})
    two_hand_frames_consistent: int = 0
    two_hand_frames_same_label: int = 0
    inference_ms: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Preparación
# --------------------------------------------------------------------------- #

def print_versions() -> None:
    print(f"Python      : {platform.python_version()} ({platform.machine()})")
    print(f"Sistema     : {platform.system()} {platform.release()}")
    print(f"MediaPipe   : {mp.__version__}")
    print(f"OpenCV      : {cv2.__version__}")
    print(f"NumPy       : {np.__version__}")


def ensure_model(path: Path) -> Path:
    """Descarga el modelo de HandLandmarker si no está en disco."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo en {path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except OSError as error:
        sys.exit(
            f"No se pudo descargar el modelo ({error}).\n"
            f"Descárgalo a mano desde:\n  {MODEL_URL}\ny guárdalo en {path}"
        )
    return path


def open_camera(device_index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(device_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        sys.exit(
            "No se pudo abrir la cámara. En macOS revisa que la app de terminal "
            "tenga permiso de cámara en Ajustes del Sistema > Privacidad y seguridad."
        )
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Cámara      : {width}x{height} @ {reported_fps:.0f} fps (según el driver)")
    return camera


def build_landmarker(model_path: Path) -> vision.HandLandmarker:
    """Umbrales explícitos, nunca los defaults de la librería."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# Lógica por frame
# --------------------------------------------------------------------------- #

def handedness_is_consistent(result: vision.HandLandmarkerResult) -> bool | None:
    """Con imagen en espejo, la mano 'Left' debería verse a la izquierda.

    Devuelve None si no hay exactamente una 'Left' y una 'Right'.
    """
    wrist_x_by_label: dict[str, float] = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        wrist_x_by_label[handedness[0].category_name] = landmarks[WRIST].x
    if set(wrist_x_by_label) != {"Left", "Right"}:
        return None
    return wrist_x_by_label["Left"] < wrist_x_by_label["Right"]


def update_stats(stats: SessionStats, result: vision.HandLandmarkerResult) -> None:
    hand_count = min(len(result.hand_landmarks), 2)
    stats.hands_count[hand_count] += 1
    if hand_count != 2:
        return
    consistent = handedness_is_consistent(result)
    if consistent is None:
        stats.two_hand_frames_same_label += 1
    elif consistent:
        stats.two_hand_frames_consistent += 1


def draw_hands(frame: np.ndarray, result: vision.HandLandmarkerResult) -> None:
    height, width = frame.shape[:2]
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name
        color = (255, 128, 0) if label == "Left" else (0, 200, 255)
        for point in landmarks:
            cv2.circle(frame, (int(point.x * width), int(point.y * height)), 3, color, -1)
        wrist = landmarks[WRIST]
        cv2.putText(
            frame, label, (int(wrist.x * width), int(wrist.y * height) + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )


def draw_overlay(frame: np.ndarray, hand_count: int, inference_ms: float) -> None:
    text = f"manos: {hand_count}  inferencia: {inference_ms:.1f} ms  (q para salir)"
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# --------------------------------------------------------------------------- #
# Resumen
# --------------------------------------------------------------------------- #

def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def print_summary(stats: SessionStats, elapsed_s: float) -> None:
    processed = stats.frames
    if processed == 0:
        print("\nNo se procesó ningún frame.")
        return

    two_hands = stats.hands_count[2]
    print("\n=========== RESUMEN ===========")
    print(f"Frames procesados : {processed} en {elapsed_s:.1f} s "
          f"-> {processed / elapsed_s:.1f} FPS reales")
    print(f"Frames perdidos   : {stats.dropped_frames}")
    print(f"Inferencia        : p50 {statistics.median(stats.inference_ms):.1f} ms | "
          f"p95 {percentile(stats.inference_ms, 0.95):.1f} ms   (objetivo p95 <= ~35 ms)")
    print(f"Manos detectadas  : 0 -> {stats.hands_count[0]}, "
          f"1 -> {stats.hands_count[1]}, 2 -> {two_hands}")
    print(f"Frames con 2 manos: {100 * two_hands / processed:.1f} %   "
          f"(solo es significativo si mantuviste las dos manos en cuadro)")
    if two_hands:
        consistent = 100 * stats.two_hand_frames_consistent / two_hands
        same_label = 100 * stats.two_hand_frames_same_label / two_hands
        print(f"Lateralidad OK    : {consistent:.1f} % de los frames con 2 manos")
        print(f"Misma etiqueta x2 : {same_label:.1f} %  (ambas 'Left' o ambas 'Right')")
        if consistent < 5:
            print("  -> Sale casi siempre al revés: la convención de espejo está "
                  "invertida respecto a lo esperado. Anótalo, no es un fallo.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0, help="índice de la cámara")
    parser.add_argument("--seconds", type=float, default=20.0, help="duración de la prueba")
    parser.add_argument("--no-mirror", action="store_true",
                        help="no voltear la imagen (para comparar la lateralidad)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    print_versions()
    model_path = ensure_model(MODEL_PATH)
    camera = open_camera(args.camera)
    landmarker = build_landmarker(model_path)
    stats = SessionStats()

    start = time.perf_counter()
    last_timestamp_ms = -1
    try:
        while time.perf_counter() - start < args.seconds:
            ok, frame = camera.read()
            if not ok or frame is None:
                stats.dropped_frames += 1
                continue
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # El modo VIDEO exige timestamps estrictamente crecientes.
            timestamp_ms = max(int((time.perf_counter() - start) * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            inference_start = time.perf_counter()
            result = landmarker.detect_for_video(image, timestamp_ms)
            inference_ms = (time.perf_counter() - inference_start) * 1000

            stats.frames += 1
            stats.inference_ms.append(inference_ms)
            update_stats(stats, result)

            draw_hands(frame, result)
            draw_overlay(frame, len(result.hand_landmarks), inference_ms)
            cv2.imshow("check_env", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed = time.perf_counter() - start
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()
        print_summary(stats, elapsed)


if __name__ == "__main__":
    run(parse_args())
