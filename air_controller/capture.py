"""Capa 1: fuentes de vídeo.

`LatestFrameCamera` lee la webcam en un hilo y se queda SOLO con el frame más
reciente. En el spike medimos ~38 ms de MediaPipe por frame y la cámara da uno
cada ~33 ms: si leyéramos en orden, los frames se acumularían en el buffer y
el retraso crecería sin parar. Tirar los frames viejos mantiene el retraso
acotado a ~1 frame.

`VideoFileSource` lee un vídeo grabado en orden, sin tirar nada, para probar
sin cámara.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray       # BGR, ya en espejo
    timestamp_ms: int       # estrictamente creciente


class FrameSource(Protocol):
    def read(self) -> Frame | None:
        """Siguiente frame, o None si la fuente se ha acabado."""

    def close(self) -> None: ...


class LatestFrameCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._capture = cv2.VideoCapture(index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError("No se pudo abrir la cámara. Revisa los permisos de macOS.")
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._latest: Frame | None = None
        self._running = True
        self._last_timestamp_ms = -1
        self._start = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, image = self._capture.read()
            if not ok:
                continue
            timestamp_ms = max(int((time.perf_counter() - self._start) * 1000),
                               self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            with self._lock:
                self._latest = Frame(cv2.flip(image, 1), timestamp_ms)
            self._new_frame.set()

    def read(self) -> Frame | None:
        """Espera a que haya un frame nuevo y devuelve el más reciente."""
        if not self._new_frame.wait(timeout=2.0):
            raise RuntimeError("La cámara no envía frames.")
        with self._lock:
            self._new_frame.clear()
            return self._latest

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._capture.release()


class VideoFileSource:
    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el vídeo {path}")
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok:
            return None
        frame = Frame(cv2.flip(image, 1), int(self._index * 1000 / self._fps))
        self._index += 1
        return frame

    def close(self) -> None:
        self._capture.release()
