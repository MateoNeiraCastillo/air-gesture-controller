"""Spike step 1: records raw webcam video with a label.

Records WITHOUT flipping and WITHOUT MediaPipe: the video is the raw material
and can be reprocessed with any parameters. The preview is mirrored so it
feels natural.

Usage:
    python scripts/record.py --label rest
    python scripts/record.py --label a_held --seconds 30 --light night
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
LABELS = ("rest", "a_held", "a_rhythm", "dpad_left", "dpad_right", "combo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Records a labelled clip for the spike.")
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--light", default="night", help="lighting condition, e.g. day / night")
    return parser.parse_args()


def next_output_path(label: str, light: str) -> Path:
    """data/raw/<label>_<light>_<NN>.mp4 without overwriting earlier recordings."""
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
        raise RuntimeError("Could not open the camera.")
    return camera


def show_preview(frame, text: str, color: tuple[int, int, int]) -> bool:
    """Shows the mirrored frame with some text. Returns True if 'q' was pressed."""
    preview = cv2.flip(frame, 1)
    cv2.putText(preview, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("record", preview)
    return cv2.waitKey(1) & 0xFF == ord("q")


def countdown(camera: cv2.VideoCapture, label: str) -> bool:
    """Countdown to position your hands (not recorded). False if cancelled."""
    end = time.perf_counter() + COUNTDOWN_SECONDS
    while (remaining := end - time.perf_counter()) > 0:
        ok, frame = camera.read()
        if ok and show_preview(frame, f"{label}: starting in {remaining:.0f}", (0, 200, 255)):
            return False
    return True


def record(camera: cv2.VideoCapture, path: Path, seconds: float) -> int | None:
    """Records up to `seconds`. If 'q' is pressed, deletes the take and returns None:
    a half-finished clip would contaminate the analysis."""
    ok, frame = camera.read()
    if not ok:
        raise RuntimeError("The camera is not returning frames.")
    height, width = frame.shape[:2]
    fps = camera.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames_written = 0
    cancelled = False
    start = time.perf_counter()
    try:
        while (elapsed := time.perf_counter() - start) < seconds:
            ok, frame = camera.read()
            if not ok:
                continue
            writer.write(frame)
            frames_written += 1
            if show_preview(frame, f"REC {elapsed:4.1f}/{seconds:.0f} s", (0, 0, 255)):
                cancelled = True
                break
    finally:
        writer.release()
    if cancelled:
        path.unlink(missing_ok=True)
        return None
    return frames_written


def main() -> None:
    args = parse_args()
    path = next_output_path(args.label, args.light)
    camera = open_camera(CAMERA_INDEX)
    try:
        if not countdown(camera, args.label):
            print("Cancelled, nothing was saved.")
            return
        frames = record(camera, path, args.seconds)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    if frames is None:
        print("Take cancelled with 'q': nothing was saved.")
        return
    print(f"Saved {path.name}: {frames} frames")


if __name__ == "__main__":
    main()
