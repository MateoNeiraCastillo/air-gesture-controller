"""Does a simple threshold separate two distributions? The spike's core metric."""

from dataclasses import dataclass

import numpy as np

MAX_CANDIDATES = 500


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    balanced_accuracy: float
    positive_above: bool  # True: values > threshold are classified as the "positive" class


def best_threshold(negative: np.ndarray, positive: np.ndarray) -> ThresholdResult:
    """Threshold that best separates `negative` from `positive`, by balanced accuracy.

    Balanced accuracy is the mean of the per-class accuracy, so it doesn't
    favour the class with more frames. 0.5 = chance, 1.0 = perfect separation.
    NaNs (hand not detected) are discarded.
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
        below = 1 - above  # same rule with the direction flipped
        for accuracy, positive_above in ((above, True), (below, False)):
            if accuracy > best.balanced_accuracy:
                best = ThresholdResult(float(threshold), float(accuracy), positive_above)
    return best
