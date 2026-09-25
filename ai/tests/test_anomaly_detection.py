import numpy as np
import pandas as pd
import pytest

from ai.src.anomaly_detection import AnomalyDetector


FEATURES = [
    "RMS",
    "Peak",
    "Std",
    "Kurtosis",
    "Dominant_Frequency",
]


def make_healthy_data(
    rows: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic healthy feature data for testing."""

    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "RMS": rng.normal(0.20, 0.02, rows),
            "Peak": rng.normal(0.45, 0.04, rows),
            "Std": rng.normal(0.10, 0.01, rows),
            "Kurtosis": rng.normal(3.0, 0.20, rows),
            "Dominant_Frequency": rng.normal(150.0, 3.0, rows),
        }
    )


def make_anomalous_data(
    rows: int = 10,
    seed: int = 123,
) -> pd.DataFrame:
    """Generate synthetic anomalous feature data for testing."""

    rng = np.random.default_rng(seed)

    return pd.DataFrame(
        {
            "RMS": rng.normal(0.80, 0.05, rows),
            "Peak": rng.normal(1.50, 0.10, rows),
            "Std": rng.normal(0.40, 0.03, rows),
            "Kurtosis": rng.normal(7.0, 0.50, rows),
            "Dominant_Frequency": rng.normal(300.0, 10.0, rows),
        }
    )


def test_detector_can_fit() -> None:
    healthy = make_healthy_data()

    detector = AnomalyDetector(
        n_estimators=100,
        random_state=42,
    )

    result = detector.fit(healthy)

    assert result is detector


def test_detector_predicts_normal_data() -> None:
    healthy = make_healthy_data()

    detector = AnomalyDetector(
        n_estimators=100,
        random_state=42,
    )

    detector.fit(healthy)

    prediction = detector.predict(healthy)

    assert len(prediction.labels) == len(healthy)
    assert len(prediction.scores) == len(healthy)


def test_detector_can_detect_anomalous_data() -> None:
    healthy = make_healthy_data()
    anomalous = make_anomalous_data()

    detector = AnomalyDetector(
        n_estimators=200,
        contamination=0.10,
        random_state=42,
    )

    detector.fit(healthy)

    prediction = detector.predict(anomalous)

    anomaly_ratio = prediction.anomaly_count / len(anomalous)

    assert anomaly_ratio >= 0.5


def test_model_rejects_non_numeric_features() -> None:
    data = make_healthy_data()

    data["RMS"] = ["bad"] * len(data)

    detector = AnomalyDetector()

    with pytest.raises(TypeError):
        detector.fit(data)


def test_model_rejects_nan_values() -> None:
    data = make_healthy_data()

    data.loc[0, "RMS"] = np.nan

    detector = AnomalyDetector()

    with pytest.raises(ValueError):
        detector.fit(data)


def test_feature_schema_mismatch_is_detected() -> None:
    healthy = make_healthy_data()

    detector = AnomalyDetector()

    detector.fit(healthy)

    wrong_schema = healthy[
        [
            "Peak",
            "RMS",
            "Std",
            "Kurtosis",
            "Dominant_Frequency",
        ]
    ]

    with pytest.raises(ValueError):
        detector.predict(wrong_schema)


def test_unfitted_detector_cannot_predict() -> None:
    healthy = make_healthy_data()

    detector = AnomalyDetector()

    with pytest.raises(RuntimeError):
        detector.predict(healthy)


def test_model_save_and_load(tmp_path) -> None:
    healthy = make_healthy_data()

    detector = AnomalyDetector(
        n_estimators=100,
        random_state=42,
    )

    detector.fit(healthy)

    model_path = tmp_path / "anomaly_detector.joblib"

    detector.save(model_path)

    loaded_detector = AnomalyDetector.load(model_path)

    original = detector.predict(healthy)
    loaded = loaded_detector.predict(healthy)

    np.testing.assert_array_equal(
        original.labels,
        loaded.labels,
    )

    np.testing.assert_allclose(
        original.scores,
        loaded.scores,
    )