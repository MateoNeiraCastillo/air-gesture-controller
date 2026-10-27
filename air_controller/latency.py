"""Medición de latencia del bucle, por etapas.

Para cada frame apunta cuánto tarda cada etapa (en ms) y, cuando se envía una
tecla, la latencia de punta a punta: desde que la cámara entregó el frame
hasta que la tecla sale hacia el juego.

Lo que NO mide: el retraso interno de la cámara (exposición + USB, ~30–60 ms
típicos en webcams) ni lo que tarde el navegador en pintar. Es la parte que
controla nuestro código, que es la que podemos mejorar.
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
        lines = ["", "== Latencia (ms) ==", f"{'etapa':<22}{'p50':>8}{'p95':>8}{'max':>8}"]
        for name, values in self.stages.items():
            lines.append(_row(name, values))
        if self.end_to_end_ms:
            lines.append(_row("frame → tecla", self.end_to_end_ms))
        else:
            lines.append("frame → tecla: sin teclas enviadas")
        lines.append(f"FPS del bucle: {self.fps():.1f}")
        return "\n".join(lines)


def _row(name: str, values: list[float]) -> str:
    if not values:
        return f"{name:<22}{'-':>8}{'-':>8}{'-':>8}"
    array = np.array(values)
    return (f"{name:<22}{np.percentile(array, 50):>8.1f}{np.percentile(array, 95):>8.1f}"
            f"{array.max():>8.1f}")
