import numpy as np
import pytest

from air_controller.centroid import UNKNOWN, CentroidClassifier


def test_assigns_nearest_class() -> None:
    classifier = CentroidClassifier.fit({
        "rest": np.array([[0.0, 0.0], [0.1, -0.1]]),
        "pressed": np.array([[5.0, 5.0], [5.1, 4.9]]),
    })
    predicted = classifier.predict(np.array([[0.2, 0.0], [4.8, 5.2]]))
    assert list(predicted) == ["rest", "pressed"]


def test_scale_does_not_dominate() -> None:
    # Feature 0 has a huge scale but doesn't separate; feature 1 is small and does.
    classifier = CentroidClassifier.fit({
        "a": np.array([[1000.0, 0.0], [-1000.0, 0.0]]),
        "b": np.array([[1000.0, 1.0], [-1000.0, 1.0]]),
    })
    assert list(classifier.predict(np.array([[900.0, 0.1], [-900.0, 0.9]]))) == ["a", "b"]


def test_nan_rows_are_unknown_and_ignored_in_fit() -> None:
    classifier = CentroidClassifier.fit({
        "a": np.array([[0.0, 0.0], [np.nan, 1.0]]),
        "b": np.array([[1.0, 1.0]]),
    })
    predicted = classifier.predict(np.array([[np.nan, 0.0], [0.0, 0.1]]))
    assert predicted[0] == UNKNOWN
    assert predicted[1] == "a"


def test_class_without_valid_samples_raises() -> None:
    with pytest.raises(ValueError):
        CentroidClassifier.fit({"a": np.array([[np.nan, np.nan]]), "b": np.array([[1.0, 1.0]])})
