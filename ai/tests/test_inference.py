import numpy as np
import pandas as pd
import pytest

from ai.src.anomaly_detection import AnomalyDetector
from ai.src.degradation import DegradationAnalyzer, WarningLevel
from ai.src.inference import InferenceResult, VibrationInferenceEngine


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
    assert result.degradation is not None


def test_degradation_result_is_returned(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    result = trained_engine.predict(healthy)

    assert result.degradation.current_score == pytest.approx(
        result.anomaly.scores[-1]
    )

    assert isinstance(
        result.degradation.warning_level,
        WarningLevel,
    )


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
    assert result.degradation is not None


def test_score_history_accumulates(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    trained_engine.predict(healthy)
    trained_engine.predict(healthy)

    assert len(trained_engine.score_history) == 20


def test_score_history_is_read_only(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    trained_engine.predict(healthy)

    history = trained_engine.score_history

    assert isinstance(history, tuple)
    assert len(history) == 10


def test_reset_history(
    trained_engine: VibrationInferenceEngine,
) -> None:
    healthy = make_healthy_data(rows=10)

    trained_engine.predict(healthy)

    assert len(trained_engine.score_history) == 10

    trained_engine.reset_history()

    assert len(trained_engine.score_history) == 0


def test_custom_degradation_analyzer(
    trained_engine: VibrationInferenceEngine,
) -> None:
    analyzer = DegradationAnalyzer(
        moving_average_window=3,
    )

    detector = trained_engine.anomaly_detector

    engine = VibrationInferenceEngine(
        detector,
        degradation_analyzer=analyzer,
    )

    healthy = make_healthy_data(rows=10)

    result = engine.predict(healthy)

    assert result.degradation is not None
    assert len(engine.score_history) == 10


def test_invalid_degradation_analyzer() -> None:
    detector = AnomalyDetector()

    with pytest.raises(TypeError):
        VibrationInferenceEngine(
            detector,
            degradation_analyzer="invalid",
        )