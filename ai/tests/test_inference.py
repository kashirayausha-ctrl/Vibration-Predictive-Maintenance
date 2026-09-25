import numpy as np
import pandas as pd
import pytest

from ai.src.anomaly_detection import AnomalyDetector
from ai.src.inference import InferenceResult, VibrationInferenceEngine


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


@pytest.fixture
def trained_engine() -> VibrationInferenceEngine:
    healthy = make_healthy_data()

    detector = AnomalyDetector(
        n_estimators=100,
        random_state=42,
    )

    detector.fit(healthy)

    return VibrationInferenceEngine(detector)


def test_engine_requires_anomaly_detector() -> None:
    with pytest.raises(TypeError):
        VibrationInferenceEngine("not-a-detector")


def test_engine_returns_inference_result(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    result = trained_engine.predict(healthy)

    assert isinstance(result, InferenceResult)
    assert len(result.anomaly.labels) == 10
    assert len(result.anomaly.scores) == 10


def test_anomaly_ratio_is_valid(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    result = trained_engine.predict(healthy)

    assert 0.0 <= result.anomaly_ratio <= 1.0


def test_is_anomalous_returns_boolean(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    result = trained_engine.predict(healthy)

    assert isinstance(result.is_anomalous, bool)


def test_anomalous_data_can_reach_inference_layer(
    trained_engine: VibrationInferenceEngine,
) -> None:
    anomalous = make_anomalous_data(rows=10)

    result = trained_engine.predict(anomalous)

    assert len(result.anomaly.labels) == 10
    assert len(result.anomaly.scores) == 10