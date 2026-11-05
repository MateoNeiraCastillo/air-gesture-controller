"""Spike step 3: does the invisible controller work, and how much does it change between sessions?

Reads data/landmarks/*.npz (name: <label>_<session>_<NN>.npz), classifies with
the thumb position and a centroid classifier, and measures:

  1. Detection per clip.
  2. Per session, with CALIBRATION: the first CALIBRATION_FRAMES valid frames
     of each clip place the centroids; the rest of the clip is used to evaluate.
  3. Across sessions (if there are 2 or more): calibrate in one session and use
     it in another, without recalibrating. If this drops a lot, calibrating at
     startup is mandatory.

Usage:
    python scripts/analyze.py
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # we only save PNGs, no windows are opened
import matplotlib.pyplot as plt
import numpy as np

from air_controller.centroid import CentroidClassifier
from air_controller.features import thumb_position_features
from air_controller.recordings import parse_clip_name

ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = ROOT / "data" / "landmarks"
REPORT_DIR = ROOT / "data" / "report"

SETTLE_SECONDS = 1.0        # the first second of each clip is discarded
CALIBRATION_FRAMES = 90     # ~3 s of VALID frames (hand detected) per pose
# Columns of thumb_position_features: 0 dx_index, 1 dy_index, 2 dx_wrist, 3 dy_wrist.
# Button A: only the tip height. The dx values change if the hand rotates slightly
# between clips (seen in the spike) and dragged the classifier off.
BUTTON_FEATURES = [1, 3]
DPAD_FEATURES = [0, 1, 2, 3]
SMOOTHING_FRAMES = 3        # moving median when counting presses

TWO_HANDS_TARGET = 95.0
ACCURACY_TARGET = 95.0
INFERENCE_P95_TARGET_MS = 35.0

BUTTON_LABELS = ("rest", "a_held")
DPAD_LABELS = ("rest", "dpad_left", "dpad_right")


@dataclass(frozen=True)
class Clip:
    label: str
    session: str
    fps: float
    left: np.ndarray            # (frames, 4) left thumb features
    right: np.ndarray           # (frames, 4) right thumb features
    hands_detected: np.ndarray
    inference_ms: np.ndarray

    def features(self, hand: str, columns: list[int]) -> np.ndarray:
        return (self.left if hand == "left" else self.right)[:, columns]

    def split(self, hand: str, columns: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(calibration, evaluation): the first CALIBRATION_FRAMES valid frames
        calibrate; everything after evaluates. As in the real app, calibration
        waits to see the hand instead of using fixed seconds."""
        features = self.features(hand, columns)
        valid = np.flatnonzero(~np.isnan(features).any(axis=1))
        if len(valid) <= CALIBRATION_FRAMES:
            return features[valid], features[:0]
        cut = valid[CALIBRATION_FRAMES]
        return features[valid[:CALIBRATION_FRAMES]], features[cut:]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def features_per_frame(hands: np.ndarray) -> np.ndarray:
    return np.array([thumb_position_features(hand) for hand in hands]).reshape(-1, 4)


def load_clips() -> dict[str, dict[str, Clip]]:
    """session → label → Clip. If there are several identical takes, the last one is used."""
    sessions: dict[str, dict[str, Clip]] = defaultdict(dict)
    for path in sorted(LANDMARKS_DIR.glob("*.npz")):
        label, session, _index = parse_clip_name(path.stem)
        with np.load(path) as data:
            fps = float(data["fps"])
            settle = int(SETTLE_SECONDS * fps)
            sessions[session][label] = Clip(
                label=label,
                session=session,
                fps=fps,
                left=features_per_frame(data["left"][settle:]),
                right=features_per_frame(data["right"][settle:]),
                hands_detected=data["hands_detected"][settle:],
                inference_ms=data["inference_ms"][settle:],
            )
    return dict(sessions)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def accuracy(classifier: CentroidClassifier, features: np.ndarray, expected: str) -> float:
    """% of valid frames classified as `expected` (NaN = hand not seen, doesn't count)."""
    predicted = classifier.predict(features)
    valid = predicted != ""
    return 100 * float(np.mean(predicted[valid] == expected)) if valid.any() else float("nan")


def balanced_accuracy(classifier: CentroidClassifier, clips: dict[str, Clip],
                      labels: tuple[str, ...], hand: str, columns: list[int],
                      after_calibration: bool) -> float:
    """Mean per-class accuracy. after_calibration=True: only frames after the
    calibration (same session); False: the whole clip (another session)."""
    scores = []
    for label in labels:
        clip = clips[label]
        features = clip.split(hand, columns)[1] if after_calibration else clip.features(hand, columns)
        scores.append(accuracy(classifier, features, label))
    return float(np.nanmean(scores))


def calibrate(clips: dict[str, Clip], labels: tuple[str, ...], hand: str,
              columns: list[int]) -> CentroidClassifier:
    return CentroidClassifier.fit({label: clips[label].split(hand, columns)[0] for label in labels})


def smooth_labels(predicted: np.ndarray, positive: str, frames: int) -> np.ndarray:
    """Boolean 'is positive' with a moving median (removes 1-frame flickers)."""
    is_positive = (predicted == positive).astype(float)
    pad = frames // 2
    padded = np.pad(is_positive, pad, mode="edge")
    return np.array([np.median(padded[i:i + frames]) for i in range(len(is_positive))]) > 0.5


def count_presses(pressed: np.ndarray) -> int:
    return int(np.sum(pressed[1:] & ~pressed[:-1]))


def has_all(clips: dict[str, Clip], labels: tuple[str, ...]) -> bool:
    return all(label in clips for label in labels)


# --------------------------------------------------------------------------- #
# Report sections
# --------------------------------------------------------------------------- #

def report_detection(sessions: dict[str, dict[str, Clip]]) -> tuple[float, float]:
    print("\n== Detection per clip ==")
    print(f"{'session':<10}{'label':<20}{'frames':>8}{'2 hands':>10}{'inf p50':>10}{'inf p95':>10}")
    counts, times = [], []
    for session, clips in sorted(sessions.items()):
        for label, clip in sorted(clips.items()):
            counts.append(clip.hands_detected)
            times.append(clip.inference_ms)
            print(f"{session:<10}{label:<20}{len(clip.hands_detected):>8}"
                  f"{100 * np.mean(clip.hands_detected == 2):>9.1f}%"
                  f"{np.median(clip.inference_ms):>8.1f}ms{np.percentile(clip.inference_ms, 95):>8.1f}ms")
    all_counts = np.concatenate(counts)
    all_times = np.concatenate(times)
    return 100 * float(np.mean(all_counts == 2)), float(np.percentile(all_times, 95))


def report_session(session: str, clips: dict[str, Clip]) -> tuple[float, float]:
    """Calibrates with the first seconds and evaluates with the rest. Returns (button, dpad)."""
    print(f"\n== Session '{session}' (calibration: first {CALIBRATION_FRAMES} valid frames) ==")
    button = dpad = float("nan")

    if has_all(clips, BUTTON_LABELS):
        classifier = calibrate(clips, BUTTON_LABELS, "right", BUTTON_FEATURES)
        button = balanced_accuracy(classifier, clips, BUTTON_LABELS, "right", BUTTON_FEATURES,
                                   after_calibration=True)
        print(f"  Button A  rest vs pressed:   {button:5.1f} %")
        for label in ("a_rhythm", "combo"):
            if label in clips:
                pressed = smooth_labels(classifier.predict(clips[label].features("right", BUTTON_FEATURES)),
                                        "a_held", SMOOTHING_FRAMES)
                seconds = len(pressed) / clips[label].fps
                print(f"            {label}: {count_presses(pressed)} presses in "
                      f"{seconds:.0f} s (expected ~1 per second)")
    else:
        print("  Button A: missing 'rest' or 'a_held' clips.")

    if has_all(clips, DPAD_LABELS):
        classifier = calibrate(clips, DPAD_LABELS, "left", DPAD_FEATURES)
        dpad = balanced_accuracy(classifier, clips, DPAD_LABELS, "left", DPAD_FEATURES,
                                 after_calibration=True)
        print(f"  D-pad     rest/left/right:   {dpad:5.1f} %")
        if "combo" in clips:
            right_held = accuracy(classifier, clips["combo"].features("left", DPAD_FEATURES), "dpad_right")
            print(f"            combo read as 'right': {right_held:5.1f} %  "
                  f"(pressing A must not move the other thumb)")
    else:
        print("  D-pad: missing 'rest', 'dpad_left' or 'dpad_right' clips.")
    return button, dpad


def report_cross_session(sessions: dict[str, dict[str, Clip]]) -> float:
    """Calibrate in A and use in B without recalibrating. Returns the worst case."""
    if len(sessions) < 2:
        print("\n== Across sessions ==\n  Only one session: record another with a different --light.")
        return float("nan")
    print("\n== Across sessions: calibrate in one and use in another WITHOUT recalibrating ==")
    worst = float("inf")
    for source, target in permutations(sorted(sessions), 2):
        a, b = sessions[source], sessions[target]
        parts = []
        if has_all(a, BUTTON_LABELS) and has_all(b, BUTTON_LABELS):
            score = balanced_accuracy(calibrate(a, BUTTON_LABELS, "right", BUTTON_FEATURES), b,
                                      BUTTON_LABELS, "right", BUTTON_FEATURES, after_calibration=False)
            parts.append(f"button {score:5.1f} %")
            worst = min(worst, score)
        if has_all(a, DPAD_LABELS) and has_all(b, DPAD_LABELS):
            score = balanced_accuracy(calibrate(a, DPAD_LABELS, "left", DPAD_FEATURES), b,
                                      DPAD_LABELS, "left", DPAD_FEATURES, after_calibration=False)
            parts.append(f"D-pad {score:5.1f} %")
            worst = min(worst, score)
        print(f"  {source:>10} → {target:<10} {'   '.join(parts)}")
    return worst


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_feature_space(sessions: dict[str, dict[str, Clip]], hand: str,
                       labels: tuple[str, ...], filename: str, title: str) -> None:
    """Thumb tip relative to the index finger base (dx, dy), per session and label."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "^", "s", "D"]
    for marker, (session, clips) in zip(markers, sorted(sessions.items())):
        for label in labels:
            if label not in clips:
                continue
            features = clips[label].left if hand == "left" else clips[label].right
            ax.scatter(features[:, 0], features[:, 1], s=4, alpha=0.35, marker=marker,
                       label=f"{session} · {label}")
    ax.invert_yaxis()  # in images, y grows downwards
    ax.set_xlabel("dx (hand sizes)")
    ax.set_ylabel("dy (hand sizes)")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, dpi=120)
    plt.close(fig)


def plot_rhythm(sessions: dict[str, dict[str, Clip]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for session, clips in sorted(sessions.items()):
        if "a_rhythm" in clips:
            clip = clips["a_rhythm"]
            seconds = np.arange(len(clip.right)) / clip.fps
            ax.plot(seconds, clip.right[:, 1], linewidth=1, label=session)
    ax.set_title("a_rhythm: right thumb tip height (each valley/peak = one press)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("dy (hand sizes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "a_rhythm.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #

def verdict(two_hands: float, p95: float, button: float, dpad: float, cross: float) -> None:
    def line(ok: bool, text: str) -> None:
        print(f"{'OK ' if ok else 'NO '} {text}")

    print("\n================ VERDICT ================")
    line(two_hands >= TWO_HANDS_TARGET, f"2 hands detected            {two_hands:5.1f} %  (>= {TWO_HANDS_TARGET:.0f} %)")
    line(p95 <= INFERENCE_P95_TARGET_MS, f"inference p95               {p95:5.1f} ms (<= {INFERENCE_P95_TARGET_MS:.0f} ms)")
    line(button >= ACCURACY_TARGET, f"button A, calibrated        {button:5.1f} %  (>= {ACCURACY_TARGET:.0f} %)")
    line(dpad >= ACCURACY_TARGET, f"D-pad, calibrated           {dpad:5.1f} %  (>= {ACCURACY_TARGET:.0f} %)")
    if np.isnan(cross):
        print("--  across sessions           no data (a second session is missing)")
    else:
        line(cross >= ACCURACY_TARGET, f"across sessions, no recal.  {cross:5.1f} %  (if NO: calibrate at startup)")


def main() -> None:
    sessions = load_clips()
    if not sessions:
        print(f"No landmarks in {LANDMARKS_DIR}. Run scripts/extract.py first")
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    two_hands, p95 = report_detection(sessions)
    results = [report_session(session, clips) for session, clips in sorted(sessions.items())]
    cross = report_cross_session(sessions)

    plot_feature_space(sessions, "right", BUTTON_LABELS, "a_button_space.png",
                       "Button A: right thumb tip")
    plot_feature_space(sessions, "left", DPAD_LABELS, "dpad_space.png",
                       "D-pad: left thumb tip")
    plot_rhythm(sessions)

    worst_button = float(np.nanmin([r[0] for r in results]))
    worst_dpad = float(np.nanmin([r[1] for r in results]))
    verdict(two_hands, p95, worst_button, worst_dpad, cross)
    print(f"\nPlots in {REPORT_DIR}")


if __name__ == "__main__":
    main()
