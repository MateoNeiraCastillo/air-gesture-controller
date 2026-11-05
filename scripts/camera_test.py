"""Step 2: opens the webcam, shows the mirrored image and measures the real FPS."""

import time

import cv2

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def open_camera(index: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Check the macOS permissions.")
    return camera


def main() -> None:
    camera = open_camera(CAMERA_INDEX)
    frame_count = 0
    start = time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Frame dropped")
                continue
            frame = cv2.flip(frame, 1)
            frame_count += 1

            elapsed = time.perf_counter() - start
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("camera_test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"{frame_count} frames in {elapsed:.1f} s -> {frame_count / elapsed:.1f} FPS")


if __name__ == "__main__":
    main()