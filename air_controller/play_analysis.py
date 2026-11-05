"""What went wrong in a real game: diagnosis from a PlayLog file, with no script to compare to.

Each check answers one question a player would ask when "it didn't detect well":

- Was my hand lost?          → visibility and the longest gaps.
- Did quick taps get lost?   → raw pulses that never became a confirmed press
                               (shorter than the press confirmation).
- Did a held button drop out? → confirmed presses shorter than a real press,
                               and holds that were cut and restarted at once.
- Was the D-pad unstable?    → time with the thumb near the threshold, where a
                               small wobble flips it on and off.
- Did A and B get confused?  → the raw reading jumping straight from one to the other.
- Was it slow?               → frame rate and gaps between frames.

Pure functions over the arrays in the file, so they can be tested without a camera.
"""

from dataclasses import dataclass

import numpy as np

from air_controller.buttons import Button
from air_controller.guided import BUTTON_BITS

SHORT_PRESS_MS = 80          # a confirmed press this short is almost never on purpose
RESTART_GAP_MS = 120         # released and pressed again this fast: the hold was cut
SLOW_FRAME_MS = 100          # a gap between frames this long is a visible stutter
NEAR_THRESHOLD = 0.1         # D-pad progress within ±0.1 of the threshold


@dataclass(frozen=True)
class Run:
    start: int               # first frame index
    end: int                 # last frame index (inclusive)


def runs(flags: np.ndarray) -> list[Run]:
    """Consecutive stretches of True."""
    padded = np.concatenate([[False], flags.astype(bool), [False]])
    changes = np.flatnonzero(np.diff(padded.astype(int)))
    return [Run(int(a), int(b) - 1) for a, b in zip(changes[::2], changes[1::2])]


@dataclass(frozen=True)
class ButtonReport:
    button: Button
    raw_pulses: int           # stretches where the classifier saw it pressed
    confirmed: int            # presses that reached the game
    lost_taps: int            # raw pulses that never reached the game
    short_presses: int        # confirmed presses shorter than SHORT_PRESS_MS
    cut_holds: int            # released and pressed again within RESTART_GAP_MS
    median_delay_ms: float    # first raw frame → confirmed press


def button_report(button: Button, timestamps: np.ndarray, raw: np.ndarray, held: np.ndarray,
                  event_times: np.ndarray, event_buttons: np.ndarray,
                  event_pressed: np.ndarray) -> ButtonReport:
    bit = BUTTON_BITS[button]
    raw_on = (raw & bit) > 0
    held_on = (held & bit) > 0
    raw_runs = runs(raw_on)
    lost = sum(1 for run in raw_runs if not held_on[run.start:run.end + 2].any())

    mine = event_buttons == button.value
    presses = event_times[mine & event_pressed]
    releases = event_times[mine & ~event_pressed]
    durations = [releases[releases >= t][0] - t for t in presses if (releases >= t).any()]
    gaps = [presses[presses > t][0] - t for t in releases if (presses > t).any()]

    delays = []                       # only for raw pulses that did reach the game
    for run in raw_runs:
        start_ms = timestamps[run.start]
        end_ms = timestamps[min(run.end + 1, len(timestamps) - 1)]
        sent = presses[(presses >= start_ms) & (presses <= end_ms)]
        if len(sent):
            delays.append(sent[0] - start_ms)
    return ButtonReport(
        button=button, raw_pulses=len(raw_runs), confirmed=len(presses), lost_taps=lost,
        short_presses=sum(1 for d in durations if d < SHORT_PRESS_MS),
        cut_holds=sum(1 for g in gaps if g < RESTART_GAP_MS),
        median_delay_ms=float(np.median(delays)) if delays else float("nan"),
    )


@dataclass(frozen=True)
class HandReport:
    visible_pct: float
    losses: int               # times the hand disappeared
    longest_gap_ms: float


def hand_report(timestamps: np.ndarray, visible: np.ndarray) -> HandReport:
    gaps = runs(~visible.astype(bool))
    longest = max((timestamps[min(g.end + 1, len(timestamps) - 1)] - timestamps[g.start] for g in gaps),
                  default=0)
    return HandReport(100 * float(visible.mean()) if len(visible) else float("nan"), len(gaps), float(longest))


def dpad_near_threshold_pct(left_progress: np.ndarray, right_progress: np.ndarray,
                            threshold: float) -> float:
    """% of frames (left hand seen) with the D-pad within ±NEAR_THRESHOLD of its threshold."""
    progress = np.fmax(left_progress, right_progress)
    seen = ~np.isnan(progress)
    if not seen.any():
        return float("nan")
    near = np.abs(progress[seen] - threshold) <= NEAR_THRESHOLD
    return 100 * float(near.mean())


def a_b_switches(raw: np.ndarray) -> int:
    """Frames where the raw reading jumps straight from A to B or from B to A."""
    a = (raw & BUTTON_BITS[Button.A]) > 0
    b = (raw & BUTTON_BITS[Button.B]) > 0
    return int(np.sum((a[:-1] & b[1:]) | (b[:-1] & a[1:])))


def slow_frames(timestamps: np.ndarray) -> int:
    return int(np.sum(np.diff(timestamps) > SLOW_FRAME_MS))
