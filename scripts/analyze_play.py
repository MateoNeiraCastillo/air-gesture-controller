"""Diagnose a real game recorded by play.py: what the controller got wrong, and why.

Usage:
    python scripts/analyze_play.py                     # the most recent game
    python scripts/analyze_play.py data/play/play_XXXX.npz
"""

import argparse
from pathlib import Path

import numpy as np

from air_controller.buttons import Button
from air_controller.play_analysis import (
    NEAR_THRESHOLD,
    RESTART_GAP_MS,
    SHORT_PRESS_MS,
    SLOW_FRAME_MS,
    a_b_switches,
    button_report,
    dpad_near_threshold_pct,
    hand_report,
    slow_frames,
)

ROOT = Path(__file__).resolve().parent.parent
PLAY_DIR = ROOT / "data" / "play"


def latest_game() -> Path | None:
    games = sorted(PLAY_DIR.glob("play_*.npz"))
    return games[-1] if games else None


def report(path: Path) -> None:
    with np.load(path) as data:
        data = {key: data[key] for key in data.files}
    playing = ~data["paused"]
    t = data["timestamp_ms"][playing]
    if len(t) < 2:
        print("The game has no frames outside pause.")
        return
    seconds = (t[-1] - t[0]) / 1000
    print(f"Game: {path.name}  ({seconds:.0f} s, {len(t)} frames, {len(t) / seconds:.1f} fps)")
    detection = data["detection_ms"][playing]
    print(f"  MediaPipe: p50 {np.median(detection):.0f} ms, p95 {np.percentile(detection, 95):.0f} ms; "
          f"frames more than {SLOW_FRAME_MS} ms apart: {slow_frames(t)}")

    print("\n== Hands ==")
    for name in ("left", "right"):
        hand = hand_report(t, data[f"{name}_visible"][playing])
        print(f"  {name:<6} seen {hand.visible_pct:5.1f} % of frames, lost {hand.losses} times, "
              f"longest gap {hand.longest_gap_ms:.0f} ms")
    print(f"  posture warning on {100 * data['posture_warning'][playing].mean():.1f} % of frames")

    print("\n== Buttons ==")
    print(f"  {'button':<7}{'raw':>5}{'sent':>6}{'lost taps':>11}{'too short':>11}{'cut holds':>11}{'delay':>9}")
    reports = [button_report(button, t, data["raw_mask"][playing], data["held_mask"][playing],
                             data["event_times_ms"], data["event_buttons"], data["event_pressed"])
               for button in Button]
    for r in reports:
        print(f"  {r.button.value:<7}{r.raw_pulses:>5}{r.confirmed:>6}{r.lost_taps:>11}{r.short_presses:>11}"
              f"{r.cut_holds:>11}{r.median_delay_ms:>7.0f}ms")
    print(f"  lost taps: the classifier saw a press too short to be confirmed. "
          f"too short: sent but under {SHORT_PRESS_MS} ms. cut holds: released and pressed again "
          f"within {RESTART_GAP_MS} ms.")

    threshold = float(data["dpad_threshold"]) if "dpad_threshold" in data else 0.3
    near = dpad_near_threshold_pct(data["dpad_left_progress"][playing], data["dpad_right_progress"][playing],
                                   threshold)
    switches = a_b_switches(data["raw_mask"][playing])
    print(f"\n== Stability ==\n  D-pad within ±{NEAR_THRESHOLD} of its threshold: {near:.1f} % of frames"
          f"\n  A↔B jumps in the raw reading: {switches}")

    print("\n== What it suggests ==")
    for line in suggestions(reports, near, switches, data, playing):
        print(f"  - {line}")


def suggestions(reports, near_pct: float, switches: int, data, playing) -> list[str]:
    out = []
    for name in ("left", "right"):
        hand = hand_report(data["timestamp_ms"][playing], data[f"{name}_visible"][playing])
        if hand.visible_pct < 95:
            out.append(f"The {name} hand was lost often ({hand.visible_pct:.0f} % seen): more light, "
                       "or keep it further inside the frame.")
    for r in reports:
        if r.raw_pulses and r.lost_taps / r.raw_pulses > 0.2:
            out.append(f"{r.lost_taps} of {r.raw_pulses} {r.button.value} taps were too quick to be confirmed: "
                       "press_confirm_ms is filtering real taps.")
        if r.cut_holds >= 3:
            out.append(f"{r.button.value}: {r.cut_holds} holds were cut and restarted: the reading drops out "
                       "while holding (thumb near the edge of the pose, or the hand lost for a moment).")
    if near_pct > 15:
        out.append(f"The D-pad spent {near_pct:.0f} % of the time near its threshold: move the thumb further, "
                   "or recalibrate with smaller movements.")
    if switches >= 3:
        out.append(f"A and B were confused {switches} times: the thumb passes through one to reach the other.")
    if data["posture_warning"][playing].mean() > 0.1:
        out.append("Posture warnings were frequent: the hands drifted from the calibrated pose.")
    return out or ["Nothing stands out: if it still felt wrong, it is probably the delay."]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a recorded game.")
    parser.add_argument("game", nargs="?", type=Path, help="data/play/play_*.npz (default: the latest)")
    args = parser.parse_args()
    path = args.game or latest_game()
    if path is None:
        print(f"No games recorded in {PLAY_DIR}. Play with scripts/play.py first.")
        return
    report(path)


if __name__ == "__main__":
    main()
