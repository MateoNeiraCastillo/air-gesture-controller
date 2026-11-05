"""Layer 1: video sources.

`LatestFrameCamera` reads the webcam in a thread and keeps ONLY the most
recent frame. In the spike we measured ~38 ms of MediaPipe per frame and the
camera delivers one every ~33 ms: if we read in order, frames would pile up in
the buffer and the delay would grow without bound. Dropping old frames keeps
the delay bounded to ~1 frame.

`VideoFileSource` reads a recorded video in order, dropping nothing, for
testing without a camera.
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
    image: np.ndarray       # BGR, already mirrored
    timestamp_ms: int       # strictly increasing
    captured_s: float = 0.0 # time.perf_counter() when received from the camera (latency)


class FrameSource(Protocol):
    def read(self) -> Frame | None:
        """Next frame, or None if the source is exhausted."""

    def close(self) -> None: ...


class LatestFrameCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._capture = cv2.VideoCapture(index)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._capture.isOpened():
            raise RuntimeError("Could not open the camera. Check the macOS permissions.")
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
            captured_s = time.perf_counter()
            if not ok:
                continue
            timestamp_ms = max(int((time.perf_counter() - self._start) * 1000),
                               self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            with self._lock:
                self._latest = Frame(cv2.flip(image, 1), timestamp_ms, captured_s)
            self._new_frame.set()

    def read(self) -> Frame | None:
        """Waits for a new frame and returns the most recent one."""
        if not self._new_frame.wait(timeout=2.0):
            raise RuntimeError("The camera is not sending frames.")
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
            raise RuntimeError(f"Could not open the video {path}")
        self._fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0

    def read(self) -> Frame | None:
        ok, image = self._capture.read()
        if not ok:
            return None
        frame = Frame(cv2.flip(image, 1), int(self._index * 1000 / self._fps), time.perf_counter())
        self._index += 1
        return frame

    def close(self) -> None:
        self._capture.release()
