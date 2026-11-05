"""Per-session calibration: learn what YOUR poses look like before playing.

In the spike we saw that one session's references don't work in another (53–67 %
accuracy), so every game starts by calibrating five poses:

    1. rest         (both hands)  → reference for "nothing pressed"
    2. A pressed    (both hands)  → the left hand must stay at rest
    3. B pressed    (both hands)  → same
    4. D-pad left   (left hand)
    5. D-pad right  (left hand)

Decisions:
- VALID frames (hand detected) are counted, not seconds: if MediaPipe is slow
  to see the hand, calibration waits instead of running out of data.
- References use the median and a robust spread (MAD): an occasional wild
  landmark doesn't move the reference.
- Quality control: if two poses end up too close together, the calibration
  is rejected with a message saying what to repeat. This includes the case we
  saw in the spike (night): when pressing A, the left thumb drifted on its own
  to the "right" pose, and the controller would have pressed right by accident
  the whole time.

It is all pure logic: it takes landmarks, it never touches the camera.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from air_controller.buttons import Button
from air_controller.features import hand_size, thumb_position_features

# Columns of thumb_position_features stored for each hand (all four).
BUTTON_COLUMNS = (0, 1, 2, 3)
DPAD_COLUMNS = (0, 1, 2, 3)
# Each right-thumb button is decided on its own axis, as positions within
# BUTTON_COLUMNS: A is the thumb moving down (dy), B the thumb moving outwards (dx).
# Deciding A on all four columns was tried and rejected: in the spike's night
# clips the thumb's sideways position drifted between calibration and play, and
# because sideways noise is tiny, that drift alone made A look pressed all the
# time (a_rhythm: 0 presses instead of 30). On dy only, A gets 30 of ~32.
A_COLUMNS = (1, 3)
B_COLUMNS = (0, 2)

MAD_TO_STD = 1.4826   # converts MAD to standard deviation for normal data
MIN_SPREAD = 1e-3     # spread floor: avoids dividing by ~0


class Pose(Enum):
    REST = "rest"
    A_PRESSED = "A pressed"
    B_PRESSED = "B pressed"
    DPAD_LEFT = "D-pad left"
    DPAD_RIGHT = "D-pad right"


CALIBRATION_SEQUENCE = (Pose.REST, Pose.A_PRESSED, Pose.B_PRESSED, Pose.DPAD_LEFT, Pose.DPAD_RIGHT)
# Without B: to re-analyse the spike recordings, which don't have that pose.
# The resulting calibration doesn't detect B (Calibration.right_b = None).
SEQUENCE_WITHOUT_B = tuple(pose for pose in CALIBRATION_SEQUENCE if pose is not Pose.B_PRESSED)

# Which hands must be visible in each pose for the frame to count.
REQUIRED_HANDS = {
    Pose.REST: ("left", "right"),
    Pose.A_PRESSED: ("right", "left"),
    Pose.B_PRESSED: ("right", "left"),
    Pose.DPAD_LEFT: ("left",),
    Pose.DPAD_RIGHT: ("left",),
}


@dataclass(frozen=True)
class Reference:
    """A summarised pose: center (median) and robust spread per feature."""

    center: np.ndarray
    spread: np.ndarray

    def select(self, columns: tuple[int, ...]) -> "Reference":
        """The same pose, keeping only some of its features."""
        return Reference(self.center[list(columns)], self.spread[list(columns)])

    @classmethod
    def from_samples(cls, samples: np.ndarray) -> "Reference":
        center = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - center), axis=0)
        return cls(center=center, spread=np.maximum(MAD_TO_STD * mad, MIN_SPREAD))


@dataclass(frozen=True)
class DpadGeometry:
    """The calibrated D-pad, in a space scaled by the noise of the poses.

    - nearest(): which calibrated direction (left or right) is closest.
    - progress(): how far the thumb has moved from rest towards each one
      (0 = rest, 1 = as during calibration).
    """

    scale: np.ndarray
    rest: np.ndarray
    targets: dict[Button, np.ndarray]
    units: dict[Button, np.ndarray]
    lengths: dict[Button, float]

    @classmethod
    def from_references(cls, rest: Reference, left: Reference, right: Reference) -> "DpadGeometry":
        scale = np.sqrt(np.mean([r.spread ** 2 for r in (rest, left, right)], axis=0))
        rest_point = rest.center / scale
        targets = {Button.LEFT: left.center / scale, Button.RIGHT: right.center / scale}
        lengths = {button: float(np.linalg.norm(target - rest_point)) for button, target in targets.items()}
        units = {button: (target - rest_point) / max(lengths[button], MIN_SPREAD)
                 for button, target in targets.items()}
        return cls(scale, rest_point, targets, units, lengths)

    def progress(self, features: np.ndarray) -> dict[Button, float]:
        offset = features / self.scale - self.rest
        return {button: float(offset @ self.units[button]) / max(self.lengths[button], MIN_SPREAD)
                for button in self.targets}

    def nearest(self, features: np.ndarray) -> Button:
        point = features / self.scale
        return min(self.targets, key=lambda button: np.linalg.norm(point - self.targets[button]))


@dataclass(frozen=True)
class Calibration:
    """Everything the classifier needs to know about this session."""

    right_rest: Reference         # BUTTON_COLUMNS columns
    right_a: Reference            # BUTTON_COLUMNS columns
    left_rest: Reference          # DPAD_COLUMNS columns
    left_rest_while_a: Reference  # left thumb while the right hand presses A
    left_dpad_left: Reference     # DPAD_COLUMNS columns
    left_dpad_right: Reference    # DPAD_COLUMNS columns
    dpad: DpadGeometry
    left_hand_size: float         # pixels (median); to warn if the hand rotates or moves away
    right_hand_size: float
    # Size range (median of each pose): the hand may tilt a bit when moving
    # the thumb (in the daytime spike, the left one ranged from 77 % to 130 %).
    left_size_range: tuple[float, float] = (0.0, 0.0)
    right_size_range: tuple[float, float] = (0.0, 0.0)
    # Button B (None if calibrated without that pose, e.g. the spike clips).
    right_b: Reference | None = None
    left_rest_while_b: Reference | None = None


@dataclass(frozen=True)
class CalibrationResult:
    calibration: Calibration | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.calibration is not None


def separation(a: Reference, b: Reference) -> float:
    """Distance between two poses measured in pooled standard deviations.

    Counts how many "noise widths" lie between the centers: 1 = they overlap
    a lot, 3 or more = a simple threshold tells them apart reliably.
    """
    pooled = np.sqrt((a.spread ** 2 + b.spread ** 2) / 2)
    return float(np.linalg.norm((a.center - b.center) / pooled))


class CalibrationCollector:
    """Collects frames pose by pose. Fed one frame at a time.

    Usage:
        collector = CalibrationCollector(frames_per_pose=90)
        while not collector.done:
            collector.add(left_landmarks, right_landmarks)   # None if not visible
            ...show collector.current_pose and collector.progress...
        result = collector.build(min_separation=3.0)
    """

    def __init__(self, frames_per_pose: int,
                 sequence: tuple[Pose, ...] = CALIBRATION_SEQUENCE) -> None:
        if frames_per_pose < 1:
            raise ValueError("frames_per_pose must be >= 1")
        if sequence not in (CALIBRATION_SEQUENCE, SEQUENCE_WITHOUT_B):
            raise ValueError("Unsupported calibration sequence")
        self.frames_per_pose = frames_per_pose
        self.sequence = sequence
        self._step = 0
        self._features: dict[Pose, dict[str, list[np.ndarray]]] = {
            pose: {"left": [], "right": []} for pose in sequence
        }
        self._hand_sizes: dict[Pose, dict[str, list[float]]] = {
            pose: {"left": [], "right": []} for pose in sequence
        }

    @property
    def done(self) -> bool:
        return self._step >= len(self.sequence)

    @property
    def current_pose(self) -> Pose | None:
        return None if self.done else self.sequence[self._step]

    @property
    def progress(self) -> float:
        """Completed fraction of the current pose (0 to 1)."""
        pose = self.current_pose
        if pose is None:
            return 1.0
        hand = REQUIRED_HANDS[pose][0]
        return len(self._features[pose][hand]) / self.frames_per_pose

    def add(self, left: np.ndarray | None, right: np.ndarray | None) -> bool:
        """Adds a frame. Returns True if it counted (the required hands were visible)."""
        pose = self.current_pose
        if pose is None:
            return False
        hands = {"left": _valid_features(left), "right": _valid_features(right)}
        if any(hands[name] is None for name in REQUIRED_HANDS[pose]):
            return False
        for name in REQUIRED_HANDS[pose]:
            self._features[pose][name].append(hands[name])
            landmarks = left if name == "left" else right
            self._hand_sizes[pose][name].append(hand_size(landmarks))
        if len(self._features[pose][REQUIRED_HANDS[pose][0]]) >= self.frames_per_pose:
            self._step += 1
        return True

    def build(self, min_separation: float, dpad_threshold: float) -> CalibrationResult:
        if not self.done:
            return CalibrationResult(None, ["Calibration has not finished."])
        return build_calibration(self.samples(), self.hand_sizes(), min_separation, dpad_threshold)

    def samples(self) -> dict[Pose, dict[str, np.ndarray]]:
        """Features collected so far: pose → hand → array (frames, 4)."""
        return {pose: {hand: np.array(rows) for hand, rows in by_hand.items() if rows}
                for pose, by_hand in self._features.items()}

    def hand_sizes(self) -> dict[Pose, dict[str, np.ndarray]]:
        """Collected hand sizes: pose → hand → array of pixels."""
        return {pose: {hand: np.array(sizes) for hand, sizes in by_hand.items() if sizes}
                for pose, by_hand in self._hand_sizes.items()}


def build_calibration(samples: dict[Pose, dict[str, np.ndarray]],
                      hand_sizes: dict[Pose, dict[str, np.ndarray]],
                      min_separation: float,
                      dpad_threshold: float) -> CalibrationResult:
    """Builds the Calibration and checks its quality.

    `samples[pose][hand]`: array (frames, 4) of thumb_position_features.
    """
    button = list(BUTTON_COLUMNS)
    dpad = list(DPAD_COLUMNS)
    has_b = Pose.B_PRESSED in samples
    left_rest = Reference.from_samples(samples[Pose.REST]["left"][:, dpad])
    left_dpad_left = Reference.from_samples(samples[Pose.DPAD_LEFT]["left"][:, dpad])
    left_dpad_right = Reference.from_samples(samples[Pose.DPAD_RIGHT]["left"][:, dpad])
    calibration = Calibration(
        right_rest=Reference.from_samples(samples[Pose.REST]["right"][:, button]),
        right_a=Reference.from_samples(samples[Pose.A_PRESSED]["right"][:, button]),
        left_rest=left_rest,
        left_rest_while_a=Reference.from_samples(samples[Pose.A_PRESSED]["left"][:, dpad]),
        left_dpad_left=left_dpad_left,
        left_dpad_right=left_dpad_right,
        dpad=DpadGeometry.from_references(left_rest, left_dpad_left, left_dpad_right),
        left_hand_size=_median_size(hand_sizes, "left"),
        right_hand_size=_median_size(hand_sizes, "right"),
        left_size_range=_size_range(hand_sizes, "left"),
        right_size_range=_size_range(hand_sizes, "right"),
        right_b=Reference.from_samples(samples[Pose.B_PRESSED]["right"][:, button]) if has_b else None,
        left_rest_while_b=Reference.from_samples(samples[Pose.B_PRESSED]["left"][:, dpad]) if has_b else None,
    )
    problems = quality_problems(calibration, min_separation, dpad_threshold)
    return CalibrationResult(None if problems else calibration, problems)


def axis_progress(rest: Reference, pressed: Reference, columns: tuple[int, ...],
                  features: np.ndarray) -> float:
    """Position of `features` along rest → pressed on `columns`, in noise widths:
    0 at rest, 1 at the calibrated press, above 1 when going further."""
    cols = list(columns)
    scale = np.sqrt((rest.spread[cols] ** 2 + pressed.spread[cols] ** 2) / 2)
    origin = rest.center[cols] / scale
    axis = pressed.center[cols] / scale - origin
    return float((features[cols] / scale - origin) @ axis / max(float(axis @ axis), 1e-12))


# Pressing A must not move the right thumb outwards: B wins whenever the thumb is
# outwards (see classifier.py), so an A press that drifts outwards would become B.
A_OUTWARDS_LIMIT = 0.2


def quality_problems(calibration: Calibration, min_separation: float,
                     dpad_threshold: float) -> list[str]:
    """Messages for the player; empty list if the calibration is good."""
    checks = [
        (calibration.right_rest.select(A_COLUMNS), calibration.right_a.select(A_COLUMNS),
         "Button A: pressed and rest look too similar. Repeat, pressing down harder."),
        (calibration.left_rest, calibration.left_dpad_left,
         "D-pad left: almost the same as rest. Repeat, moving the thumb further left."),
        (calibration.left_rest, calibration.left_dpad_right,
         "D-pad right: almost the same as rest. Repeat, moving the thumb further right."),
        (calibration.left_dpad_left, calibration.left_dpad_right,
         "D-pad: left and right look too similar. Repeat, exaggerating both."),
    ]
    if calibration.right_b is not None:
        checks += [
            (calibration.right_rest.select(B_COLUMNS), calibration.right_b.select(B_COLUMNS),
             "Button B: almost the same as rest. Repeat, tilting the right thumb further outwards."),
            (calibration.right_a, calibration.right_b,
             "Buttons A and B: they look too similar. Repeat: A downwards, B outwards."),
        ]
    problems = [message for a, b, message in checks if separation(a, b) < min_separation]
    if calibration.right_b is not None:
        outwards = axis_progress(calibration.right_rest, calibration.right_b, B_COLUMNS,
                                 calibration.right_a.center)
        if outwards >= A_OUTWARDS_LIMIT:
            problems.append(f"Button A: your right thumb also moves outwards ({outwards:.0%} of the way to B), "
                            "so A could be read as B. Repeat, pressing A straight down.")
    problems.extend(_left_thumb_drift_problems(calibration, dpad_threshold))
    if min(calibration.left_hand_size, calibration.right_hand_size) <= 0:
        problems.append("Could not measure hand size. Repeat with your hands clearly visible.")
    return problems


def _median_size(hand_sizes: dict[Pose, dict[str, np.ndarray]], hand: str) -> float:
    values = [sizes[hand] for sizes in hand_sizes.values() if hand in sizes]
    return float(np.median(np.concatenate(values))) if values else 0.0


def _size_range(hand_sizes: dict[Pose, dict[str, np.ndarray]], hand: str) -> tuple[float, float]:
    medians = [float(np.median(sizes[hand])) for sizes in hand_sizes.values() if hand in sizes]
    return (min(medians), max(medians)) if medians else (0.0, 0.0)


def _left_thumb_drift_problems(calibration: Calibration, dpad_threshold: float) -> list[str]:
    """Does the left thumb unintentionally move towards a direction when pressing A or B?

    The "left at rest while pressing A (or B)" pose is required to stay below
    HALF the D-pad threshold, to leave some margin.
    """
    names = {Button.LEFT: "left", Button.RIGHT: "right"}
    while_pressing = {"A": calibration.left_rest_while_a, "B": calibration.left_rest_while_b}
    return [
        f"When pressing {name}, your left thumb moves towards '{names[button]}' "
        f"({value:.0%} of the travel). Repeat the calibration, making D-pad "
        f"{names[button]} more pronounced and keeping the left thumb still when pressing {name}."
        for name, reference in while_pressing.items() if reference is not None
        for button, value in calibration.dpad.progress(reference.center).items()
        if value >= dpad_threshold / 2
    ]


def left_thumb_drift(calibration: Calibration) -> dict[str, dict[Button, float]]:
    """Progress of the left thumb towards each direction while pressing A / B (for display)."""
    references = {"A": calibration.left_rest_while_a, "B": calibration.left_rest_while_b}
    return {name: calibration.dpad.progress(reference.center)
            for name, reference in references.items() if reference is not None}


def _valid_features(landmarks: np.ndarray | None) -> np.ndarray | None:
    if landmarks is None or np.isnan(landmarks).any():
        return None
    features = thumb_position_features(landmarks)
    return None if np.isnan(features).any() else features
