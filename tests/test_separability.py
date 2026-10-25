import math

import numpy as np

from air_controller.separability import best_threshold


def test_perfectly_separated_distributions() -> None:
    result = best_threshold(np.array([170.0, 172, 175, 178]), np.array([140.0, 142, 145]))
    assert result.balanced_accuracy == 1.0
    assert result.positive_above is False
    assert 145 <= result.threshold < 170


def test_identical_distributions_are_chance() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(160, 5, 2000)
    result = best_threshold(values[:1000], values[1000:])
    assert result.balanced_accuracy < 0.6


def test_nans_are_ignored() -> None:
    result = best_threshold(np.array([1.0, 2, np.nan]), np.array([10.0, np.nan, 11]))
    assert result.balanced_accuracy == 1.0


def test_empty_class_returns_nan() -> None:
    result = best_threshold(np.array([np.nan]), np.array([1.0]))
    assert math.isnan(result.balanced_accuracy)
