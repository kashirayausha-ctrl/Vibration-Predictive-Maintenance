import numpy as np
import pandas as pd
import pytest

from ai.src.classification import FaultClassifier, ClassificationPrediction


FEATURES = [
    "RMS",
    "Peak",
    "Std",
    "Kurtosis",
    "Dominant_Frequency",
]


def make_classification_data(
    rows_per_class: int = 40,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create synthetic data for testing the classifier."""

    rng = np.random.default_rng(seed)

    normal = pd.DataFrame(
        {
            "RMS": rng.normal(0.20, 0.015, rows_per_class),
            "Peak": rng.normal(0.45, 0.03, rows_per_class),
            "Std": rng.normal(0.10, 0.008, rows_per_class),
            "Kurtosis": rng.normal(3.0, 0.15, rows_per_class),
            "Dominant_Frequency": rng.normal(150.0, 2.0, rows_per_class),
        }
    )

    imbalance = pd.DataFrame(
        {
            "RMS": rng.normal(0.55, 0.02, rows_per_class),
            "Peak": rng.normal(1.10, 0.05, rows_per_class),
            "Std": rng.normal(0.25, 0.015, rows_per_class),
            "Kurtosis": rng.normal(4.5, 0.20, rows_per_class),
            "Dominant_Frequency": rng.normal(155.0, 2.0, rows_per_class),
        }
    )

    loose_mount = pd.DataFrame(
        {
            "RMS": rng.normal(0.85, 0.03, rows_per_class),
            "Peak": rng.normal(1.70, 0.07, rows_per_class),
            "Std": rng.normal(0.40, 0.02, rows_per_class),
            "Kurtosis": rng.normal(6.0, 0.30, rows_per_class),
            "Dominant_Frequency": rng.normal(300.0, 4.0, rows_per_class),
        }
    )

    X = pd.concat(
        [normal, imbalance, loose_mount],
        ignore_index=True,
    )

    y = pd.Series(
        ["normal"] * rows_per_class
        + ["imbalance"] * rows_per_class
        + ["loose_mount"] * rows_per_class,
        name="label",
    )

    return X, y


def test_classifier_can_fit() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    result = classifier.fit(X, y)

    assert result is classifier


def test_classifier_predicts_labels() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X, y)

    prediction = classifier.predict(X)

    assert isinstance(prediction, ClassificationPrediction)
    assert len(prediction.labels) == len(X)


def test_classifier_returns_probabilities() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X, y)

    prediction = classifier.predict(X)

    assert prediction.probabilities.shape == (
        len(X),
        3,
    )

    np.testing.assert_allclose(
        prediction.probabilities.sum(axis=1),
        np.ones(len(X)),
    )


def test_classifier_returns_confidence() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X, y)

    prediction = classifier.predict(X)

    assert len(prediction.confidence) == len(X)
    assert np.all(prediction.confidence >= 0.0)
    assert np.all(prediction.confidence <= 1.0)


def test_classifier_requires_multiple_classes() -> None:
    X, _ = make_classification_data()

    y = pd.Series(["normal"] * len(X))

    classifier = FaultClassifier()

    with pytest.raises(ValueError):
        classifier.fit(X, y)


def test_classifier_rejects_mismatched_lengths() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier()

    with pytest.raises(ValueError):
        classifier.fit(X.iloc[:-1], y)


def test_classifier_rejects_nan_features() -> None:
    X, y = make_classification_data()

    X.loc[0, "RMS"] = np.nan

    classifier = FaultClassifier()

    with pytest.raises(ValueError):
        classifier.fit(X, y)


def test_classifier_detects_schema_mismatch() -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X, y)

    wrong_schema = X[
        [
            "Peak",
            "RMS",
            "Std",
            "Kurtosis",
            "Dominant_Frequency",
        ]
    ]

    with pytest.raises(ValueError):
        classifier.predict(wrong_schema)


def test_unfitted_classifier_cannot_predict() -> None:
    X, _ = make_classification_data()

    classifier = FaultClassifier()

    with pytest.raises(RuntimeError):
        classifier.predict(X)


def test_classifier_save_and_load(tmp_path) -> None:
    X, y = make_classification_data()

    classifier = FaultClassifier(
        n_estimators=100,
        random_state=42,
    )

    classifier.fit(X, y)

    model_path = tmp_path / "fault_classifier.joblib"

    classifier.save(model_path)

    loaded = FaultClassifier.load(model_path)

    original = classifier.predict(X)
    restored = loaded.predict(X)

    np.testing.assert_array_equal(
        original.labels,
        restored.labels,
    )

    np.testing.assert_allclose(
        original.probabilities,
        restored.probabilities,
    )


def test_missing_model_file_raises_error() -> None:
    with pytest.raises(FileNotFoundError):
        FaultClassifier.load(
            "does_not_exist.joblib"
        )