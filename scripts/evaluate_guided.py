"""Analyses a guided session: did the controller do what was asked?

Usage:
    python scripts/evaluate_guided.py                  # the most recent session
    python scripts/evaluate_guided.py data/guided/session_XXXX.npz
"""

import argparse
from pathlib import Path

import numpy as np

from air_controller.buttons import Button
from air_controller.guided import BUTTON_BITS, StepScore, build_script, score_step

ROOT = Path(__file__).resolve().parent.parent
GUIDED_DIR = ROOT / "data" / "guided"
TARGET = 95.0


def latest_session() -> Path | None:
    sessions = sorted(GUIDED_DIR.glob("session_*.npz"))
    return sessions[-1] if sessions else None


def scores_for(data) -> list[StepScore]:
    script = build_script(int(data["blocks"]))
    timestamps = data["timestamp_ms"]
    held = data["held_mask"]
    a_presses = data["event_times_ms"][(data["event_buttons"] == Button.A.value) & data["event_pressed"]]
    starts = data["step_starts_ms"]
    return [score_step(block, step, int(starts[i]), timestamps, held, a_presses)
            for i, (block, step) in enumerate(script) if i < len(starts)]


def mean(values: list[float]) -> float:
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def report(path: Path) -> None:
    with np.load(path) as data:
        data = {key: data[key] for key in data.files}
    scores = scores_for(data)
    if not scores:
        print("The session has no recorded steps.")
        return
    print(f"Session: {path.name}  ({'complete' if data['completed'] else 'interrupted'}, "
          f"{len(data['timestamp_ms'])} frames)")

    # 1. Accuracy per button in hold steps (confirmed state vs requested)
    print("\n== Button state (what the game receives) vs what was asked ==")
    overall: dict[Button, float] = {}
    for button in BUTTON_BITS:
        overall[button] = mean([s.agreement[button] for s in scores])
        print(f"  {button.value:<6} {overall[button]:6.1f} % of frames correct")

    # 2. Worst steps: where it fails
    print("\n== Steps with any button below 90 % (the 12 worst) ==")
    bad = sorted(((s, b, v) for s in scores for b, v in s.agreement.items()
                  if not np.isnan(v) and v < 90), key=lambda item: item[2])
    if not bad:
        print("  None.")
    if len(bad) > 12:
        print(f"  ({len(bad)} in total)")
    for s, button, value in bad[:12]:
        print(f"  block {s.block}  {s.step.text:<38} {button.value:<6} {value:5.1f} %")

    # 3. A taps
    print("\n== A taps (requested → confirmed) ==")
    tap_steps = [s for s in scores if s.step.a_taps]
    exact = sum(s.a_presses == s.step.a_taps for s in tap_steps)
    for s in tap_steps:
        mark = "OK" if s.a_presses == s.step.a_taps else "  "
        print(f"  {mark} block {s.block}  {s.step.text:<38} {s.step.a_taps} → {s.a_presses}")
    tap_rate = 100 * exact / len(tap_steps) if tap_steps else float("nan")

    # 4. Drift: does it get worse over time?
    print("\n== Drift between blocks ==")
    steps = data["step"]
    blocks = np.array([scores[i].block if i < len(scores) else 0 for i in steps])
    for block in sorted(set(s.block for s in scores)):
        block_scores = [s for s in scores if s.block == block]
        agreement = mean([v for s in block_scores for v in s.agreement.values()])
        rest_frames = np.array([scores[i].step.text == "Rest" for i in steps]) & (blocks == block)
        drift = np.nanmedian(np.maximum(data["dpad_left_progress"][rest_frames],
                                        data["dpad_right_progress"][rest_frames])) if rest_frames.any() else np.nan
        warned = 100 * np.mean(data["posture_warning"][blocks == block])
        print(f"  block {block}: accuracy {agreement:5.1f} % | left thumb at rest: "
              f"{drift:+.0%} of the D-pad travel | posture warnings {warned:4.1f} %")

    # 5. Verdict
    worst = min(v for v in overall.values() if not np.isnan(v))
    print("\n================ VERDICT ================")
    print(f"{'OK ' if worst >= TARGET else 'NO '} worst button             {worst:5.1f} %  (>= {TARGET:.0f} %)")
    print(f"{'OK ' if tap_rate >= 80 else 'NO '} exact A taps             {tap_rate:5.1f} %  (>= 80 %)")
    print(f"     posture warnings         {100 * np.mean(data['posture_warning']):5.1f} % of the time")
    print(f"     press confirmation       {float(data['press_confirm_ms']):.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyses a guided session.")
    parser.add_argument("path", nargs="?", type=Path, help=".npz file (default: the latest)")
    args = parser.parse_args()
    path = args.path or latest_session()
    if path is None or not path.exists():
        print(f"No sessions in {GUIDED_DIR}. Record one with scripts/guided_session.py")
        return
    report(path)


if __name__ == "__main__":
    main()
