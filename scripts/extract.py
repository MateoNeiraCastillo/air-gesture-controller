"""Spike paso 2: pasa cada vídeo de data/raw por MediaPipe y guarda los landmarks.

Por cada vídeo genera data/landmarks/<nombre>.npz con:
    left, right       (frames, 21, 3) en píxeles; NaN si esa mano no se vio
    hands_detected    (frames,) cuántas manos devolvió MediaPipe
    inference_ms      (frames,) coste de detect_for_video
    + metadatos (label, delegate, versión de MediaPipe, tamaño del frame)

Por defecto usa CPU: en el spike la GPU perdía manos y duplicaba detecciones
(ver docs/spike-resultados.md).

Uso:
    python scripts/extract.py            # CPU, solo vídeos sin extraer
    python scripts/extract.py --gpu
    python scripts/extract.py --force    # re-extrae todo
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from air_controller.hands import assign_hands

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "landmarks"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
NUM_LANDMARKS = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae landmarks de los vídeos del spike.")
    parser.add_argument("--gpu", action="store_true", help="usar GPU (más rápida, menos fiable)")
    parser.add_argument("--force", action="store_true", help="re-extraer vídeos ya procesados")
    return parser.parse_args()


def build_landmarker(use_gpu: bool) -> vision.HandLandmarker:
    delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_mediapipe_image(frame: np.ndarray, use_gpu: bool) -> mp.Image:
    if use_gpu:
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def to_pixel_array(landmarks, width: int, height: int) -> np.ndarray:
    """Normalizado → píxeles. z se escala con el ancho, como indica MediaPipe."""
    return np.array([[p.x * width, p.y * height, p.z * width] for p in landmarks])


def label_and_session(video: Path) -> tuple[str, str]:
    """'boton_a_mantenido_noche_01' → ('boton_a_mantenido', 'noche').

    La sesión es la condición de luz con la que se grabó (--light en record.py).
    """
    label, session, _index = video.stem.rsplit("_", 2)
    return label, session


def extract_video(video: Path, use_gpu: bool) -> dict[str, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = build_landmarker(use_gpu)
    lefts, rights, counts, times_ms = [], [], [], []
    empty = np.full((NUM_LANDMARKS, 3), np.nan)
    width = height = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            image = to_mediapipe_image(frame, use_gpu)
            timestamp_ms = int(frame_index * 1000 / fps)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, timestamp_ms)
            times_ms.append((time.perf_counter() - t0) * 1000)

            assigned = assign_hands(result.hand_landmarks)
            lefts.append(empty if assigned.left is None
                         else to_pixel_array(assigned.left, width, height))
            rights.append(empty if assigned.right is None
                          else to_pixel_array(assigned.right, width, height))
            counts.append(len(result.hand_landmarks))
            frame_index += 1
    finally:
        capture.release()
        landmarker.close()

    return {
        "left": np.array(lefts).reshape(-1, NUM_LANDMARKS, 3),
        "right": np.array(rights).reshape(-1, NUM_LANDMARKS, 3),
        "hands_detected": np.array(counts, dtype=int),
        "inference_ms": np.array(times_ms),
        "label": np.array(label_and_session(video)[0]),
        "session": np.array(label_and_session(video)[1]),
        "delegate": np.array("GPU" if use_gpu else "CPU"),
        "mediapipe_version": np.array(mp.__version__),
        "frame_size": np.array([width, height]),
        "fps": np.array(fps),
    }


def main() -> None:
    args = parse_args()
    use_gpu = args.gpu
    if not MODEL_PATH.exists():
        print(f"Falta el modelo en {MODEL_PATH}. Descárgalo como en el Paso 3.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(RAW_DIR.glob("*.mp4"))
    if not videos:
        print(f"No hay vídeos en {RAW_DIR}. Graba primero con scripts/record.py")
        return
    for video in videos:
        output = OUT_DIR / f"{video.stem}.npz"
        if output.exists() and not args.force:
            print(f"{video.name}: ya extraído, se salta (usa --force para repetir)")
            continue
        data = extract_video(video, use_gpu)
        np.savez_compressed(output, **data)
        frames = len(data["hands_detected"])
        two = 100 * np.mean(data["hands_detected"] == 2) if frames else 0.0
        print(f"{video.name}: {frames} frames, 2 manos en {two:.1f} %")


if __name__ == "__main__":
    main()
