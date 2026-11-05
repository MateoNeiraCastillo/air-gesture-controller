"""Step 3 (latency): CPU vs GPU on the SAME videos, on your Mac.

In the spike the GPU lost the left hand and put both detections on the right
one. This script checks it on your machine and measures the real cost of each
option, to decide based on data.

Measures per video and delegate:
  - % of frames with two distinct hands
  - % of frames with duplicated detections (two "hands" on the same hand)
  - inference p50 / p95

Usage:
    python scripts/compare_delegates.py                    # all videos in data/raw
    python scripts/compare_delegates.py rest dpad_right    # only those labels
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
WRIST = 0
DUPLICATE_MAX_DISTANCE = 0.12  # wrists closer than 12 % of the width = the same hand
SETTLE_FRAMES = 30


@dataclass(frozen=True)
class DelegateResult:
    two_distinct: float
    duplicated: float
    p50_ms: float
    p95_ms: float


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
        return mp.Image(image_format=mp.ImageFormat.SRGBA,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def is_duplicate(result: vision.HandLandmarkerResult) -> bool:
    first, second = result.hand_landmarks
    return abs(first[WRIST].x - second[WRIST].x) < DUPLICATE_MAX_DISTANCE


def run(video: Path, use_gpu: bool) -> DelegateResult:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    landmarker = build_landmarker(use_gpu)
    distinct = duplicated = counted = 0
    times_ms: list[float] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = to_mediapipe_image(cv2.flip(frame, 1), use_gpu)
            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, int(frame_index * 1000 / fps))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            frame_index += 1
            if frame_index <= SETTLE_FRAMES:
                continue
            counted += 1
            times_ms.append(elapsed_ms)
            if len(result.hand_landmarks) == 2:
                if is_duplicate(result):
                    duplicated += 1
                else:
                    distinct += 1
    finally:
        capture.release()
        landmarker.close()
    if counted == 0:
        return DelegateResult(float("nan"), float("nan"), float("nan"), float("nan"))
    return DelegateResult(
        two_distinct=100 * distinct / counted,
        duplicated=100 * duplicated / counted,
        p50_ms=float(np.median(times_ms)),
        p95_ms=float(np.percentile(times_ms, 95)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compares CPU and GPU on the spike videos.")
    parser.add_argument("labels", nargs="*", help="labels to test (default: all)")
    args = parser.parse_args()

    videos = sorted(RAW_DIR.glob("*.mp4"))
    if args.labels:
        videos = [v for v in videos if v.stem.rsplit("_", 2)[0] in args.labels]
    if not videos:
        print(f"No matching videos in {RAW_DIR}")
        return

    print(f"{'video':<32}{'delegate':<10}{'2 hands':>9}{'dupl.':>9}{'p50':>9}{'p95':>9}")
    for video in videos:
        for use_gpu in (False, True):
            try:
                r = run(video, use_gpu)
            except (RuntimeError, ValueError) as error:
                print(f"{video.stem:<32}{'GPU' if use_gpu else 'CPU':<10} not available: {error}")
                continue
            print(f"{video.stem:<32}{'GPU' if use_gpu else 'CPU':<10}"
                  f"{r.two_distinct:>8.1f}%{r.duplicated:>8.1f}%{r.p50_ms:>7.1f}ms{r.p95_ms:>7.1f}ms")


if __name__ == "__main__":
    main()
