from air_controller.hands import assign_hands
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

"""Step 3: detects up to two hands and measures MediaPipe's cost per frame."""

import statistics
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
WRIST = 0
USE_GPU = True
WARMUP_SECONDS = 3.0


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Check the macOS permissions.")
    return camera


def build_landmarker(model_path: Path, use_gpu: bool) -> HandLandmarkerResult:
    delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_mediapipe_image(frame, use_gpu: bool) -> mp.Image:
    """The GPU only accepts 4 channels (RGBA); the CPU works with 3 (RGB)."""
    if use_gpu:
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def draw_hands(frame, result: HandLandmarkerResult) -> None:
    height, width = frame.shape[:2]
    assigned = assign_hands(result.hand_landmarks)
    for landmarks, label, color in (
        (assigned.left, "LEFT", (255, 128, 0)),
        (assigned.right, "RIGHT", (0, 200, 255)),
    ):
        if landmarks is None:
            continue
        for point in landmarks:
            center = (int(point.x * width), int(point.y * height))
            cv2.circle(frame, center, 3, color, -1)
        wrist = landmarks[WRIST]
        cv2.putText(frame, label, (int(wrist.x * width), int(wrist.y * height) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def print_summary(inference_times_ms: list[float], frames_counted: int,
                  frames_with_two_hands: int, use_gpu: bool) -> None:
    if not inference_times_ms or frames_counted == 0:
        print("Not enough frames (did you close it before the 3 s warm-up?).")
        return
    p95 = sorted(inference_times_ms)[int(0.95 * len(inference_times_ms))]
    print(f"Delegate: {'GPU' if use_gpu else 'CPU'}")
    print(f"Frames measured: {len(inference_times_ms)}")
    print(f"Inference: p50 {statistics.median(inference_times_ms):.1f} ms | p95 {p95:.1f} ms")
    print(f"Frames with 2 hands: {100 * frames_with_two_hands / frames_counted:.1f} %")


def main() -> None:
    camera = open_camera(CAMERA_INDEX)
    landmarker = build_landmarker(MODEL_PATH, USE_GPU)
    inference_times_ms: list[float] = []
    frames_counted = 0
    frames_with_two_hands = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            image = to_mediapipe_image(frame, USE_GPU)
            timestamp_ms = int((time.perf_counter() - start) * 1000)

            t0 = time.perf_counter()
            result = landmarker.detect_for_video(image, timestamp_ms)
            inference_ms = (time.perf_counter() - t0) * 1000

            # The first seconds don't count: the model is "warming up"
            # and you are still positioning your hands.
            if time.perf_counter() - start > WARMUP_SECONDS:
                inference_times_ms.append(inference_ms)
                frames_counted += 1
                if len(result.hand_landmarks) == 2:
                    frames_with_two_hands += 1

            draw_hands(frame, result)
            cv2.imshow("hands_test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print_summary(inference_times_ms, frames_counted, frames_with_two_hands, USE_GPU)


if __name__ == "__main__":
    main()