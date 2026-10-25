"""Clasificador por centroide: cada clase es la media de sus muestras.

Se usa para la calibración por sesión: unos segundos de cada postura bastan
para colocar los centroides, y cada frame nuevo se asigna al más cercano.
Las features se estandarizan (media 0, desviación 1) para que ninguna domine
solo por tener una escala mayor.
"""

from dataclasses import dataclass

import numpy as np

UNKNOWN = ""


@dataclass(frozen=True)
class CentroidClassifier:
    labels: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    centroids: np.ndarray  # (clases, features), en espacio estandarizado

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray]) -> "CentroidClassifier":
        """`samples`: etiqueta → array (n, features). Las filas con NaN se ignoran."""
        clean = {label: _drop_nan_rows(values) for label, values in samples.items()}
        empty = [label for label, values in clean.items() if len(values) == 0]
        if empty:
            raise ValueError(f"Sin muestras válidas para: {', '.join(empty)}")
        stacked = np.vstack(list(clean.values()))
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std[std == 0] = 1.0
        centroids = np.array([((values - mean) / std).mean(axis=0) for values in clean.values()])
        return cls(tuple(clean), mean, std, centroids)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Etiqueta por fila; UNKNOWN donde la fila tiene NaN (mano no detectada)."""
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
