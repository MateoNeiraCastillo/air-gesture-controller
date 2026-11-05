"""Loop latency measurement, per stage.

For each frame it records how long each stage takes (in ms) and, when a key
is sent, the end-to-end latency: from when the camera delivered the frame
until the key goes out to the game.

What it does NOT measure: the camera's internal delay (exposure + USB, ~30–60 ms
typical for webcams) or how long the browser takes to paint. It's the part
our code controls, which is the part we can improve.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LatencyTracker:
    stages: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    end_to_end_ms: list[float] = field(default_factory=list)
    frame_times_s: list[float] = field(default_factory=list)

    def add_stage(self, name: str, milliseconds: float) -> None:
        self.stages[name].append(milliseconds)

    def add_end_to_end(self, milliseconds: float) -> None:
        self.end_to_end_ms.append(milliseconds)

    def add_frame(self, timestamp_s: float) -> None:
        self.frame_times_s.append(timestamp_s)

    def fps(self) -> float:
        if len(self.frame_times_s) < 2:
            return float("nan")
        span = self.frame_times_s[-1] - self.frame_times_s[0]
        return (len(self.frame_times_s) - 1) / span if span > 0 else float("nan")

    def summary(self) -> str:
        lines = ["", "== Latency (ms) ==", f"{'stage':<22}{'p50':>8}{'p95':>8}{'max':>8}"]
        for name, values in self.stages.items():
            lines.append(_row(name, values))
        if self.end_to_end_ms:
            lines.append(_row("frame → key", self.end_to_end_ms))
        else:
            lines.append("frame → key: no keys sent")
        lines.append(f"Loop FPS: {self.fps():.1f}")
        return "\n".join(lines)


def _row(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:<22}{'-':>8}{'-':>8}{'-':>8}"
    array = np.array(values)
    return (f"{name:<22}{np.percentile(array, 50):>8.1f}{np.percentile(array, 95):>8.1f}"
            f"{array.max():>8.1f}")
