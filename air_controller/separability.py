"""¿Separa un umbral simple dos distribuciones? Métrica central del spike."""

from dataclasses import dataclass

import numpy as np

MAX_CANDIDATES = 500


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    balanced_accuracy: float
    positive_above: bool  # True: valores > umbral se clasifican como la clase "positive"


def best_threshold(negative: np.ndarray, positive: np.ndarray) -> ThresholdResult:
    """Umbral que mejor separa `negative` de `positive`, por accuracy balanceada.

    La accuracy balanceada es la media del acierto en cada clase, así que no
    favorece a la clase con más frames. 0.5 = azar, 1.0 = separación perfecta.
    Los NaN (mano no detectada) se descartan.
    """
    negative = negative[~np.isnan(negative)]
    positive = positive[~np.isnan(positive)]
    if negative.size == 0 or positive.size == 0:
        return ThresholdResult(threshold=float("nan"), balanced_accuracy=float("nan"),
                               positive_above=True)

    combined = np.concatenate([negative, positive])
    candidates = np.unique(np.quantile(combined, np.linspace(0, 1, MAX_CANDIDATES)))

    best = ThresholdResult(threshold=float("nan"), balanced_accuracy=-1.0, positive_above=True)
    for threshold in candidates:
        positive_hit = np.mean(positive > threshold)
        negative_hit = np.mean(negative <= threshold)
        above = (positive_hit + negative_hit) / 2
        below = 1 - above  # misma regla con el sentido invertido
        for accuracy, positive_above in ((above, True), (below, False)):
            if accuracy > best.balanced_accuracy:
                best = ThresholdResult(float(threshold), float(accuracy), positive_above)
    return best
