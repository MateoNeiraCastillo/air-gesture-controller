"""Centroid classifier: each class is the mean of its samples.

Used for per-session calibration: a few seconds of each pose are enough to
place the centroids, and each new frame is assigned to the nearest one.
Features are standardized (mean 0, deviation 1) so that none dominates just
because it has a larger scale.
"""

from dataclasses import dataclass

import numpy as np

UNKNOWN = ""


@dataclass(frozen=True)
class CentroidClassifier:
    labels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    centroids: np.ndarray  # (classes, features), in standardized space

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray]) -> "CentroidClassifier":
        """`samples`: label → array (n, features). Rows with NaN are ignored."""
        clean = {label: _drop_nan_rows(values) for label, values in samples.items()}
        empty = [label for label, values in clean.items() if len(values) == 0]
        if empty:
            raise ValueError(f"No valid samples for: {', '.join(empty)}")
        stacked = np.vstack(list(clean.values()))
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std == 0] = 1.0
        centroids = np.array([((values - mean) / std).mean(axis=0) for values in clean.values()])
        return cls(tuple(clean), mean, std, centroids)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Label per row; UNKNOWN where the row has NaN (hand not detected)."""
        features = np.atleast_2d(features)
        result = np.full(len(features), UNKNOWN, dtype=object)
        valid = ~np.isnan(features).any(axis=1)
        if valid.any():
            standardized = (features[valid] - self.mean) / self.std
            distances = np.linalg.norm(standardized[:, None, :] - self.centroids[None], axis=2)
            result[valid] = np.array(self.labels, dtype=object)[distances.argmin(axis=1)]
        return result


def _drop_nan_rows(values: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(values)
    return values[~np.isnan(values).any(axis=1)]
