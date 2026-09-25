import numpy as np
import pytest

from ai.src.degradation import (
    DegradationAnalyzer,
    WarningLevel,
)


def test_healthy_scores():
    analyzer = DegradationAnalyzer()

    result = analyzer.analyze(
        [0.01, 0.02, 0.01, 0.02, 0.01]
    )

    assert result.warning_level == WarningLevel.HEALTHY
    assert result.is_degrading is False


def test_monitor_warning():
    analyzer = DegradationAnalyzer(
        monitor_threshold=0.10,
        early_warning_threshold=0.25,
        high_risk_threshold=0.50,
    )

    result = analyzer.analyze(
        [0.05, 0.08, 0.12]
    )

    assert result.warning_level == WarningLevel.MONITOR


def test_early_warning():
    analyzer = DegradationAnalyzer()

    result = analyzer.analyze(
        [0.15, 0.20, 0.28]
    )

    assert result.warning_level == WarningLevel.EARLY_WARNING


def test_high_risk():
    analyzer = DegradationAnalyzer()

    result = analyzer.analyze(
        [0.30, 0.40, 0.55]
    )

    assert result.warning_level == WarningLevel.HIGH_RISK


def test_increasing_trend_detected():
    analyzer = DegradationAnalyzer(
        trend_threshold=0.01
    )

    result = analyzer.analyze(
        [0.05, 0.10, 0.20, 0.30, 0.40]
    )

    assert result.trend_slope > 0.01
    assert result.is_degrading is True


def test_decreasing_trend_not_degrading():
    analyzer = DegradationAnalyzer(
        trend_threshold=0.01
    )

    result = analyzer.analyze(
        [0.40, 0.30, 0.20, 0.10, 0.05]
    )

    assert result.trend_slope < 0
    assert result.is_degrading is False


def test_moving_average_uses_recent_window():
    analyzer = DegradationAnalyzer(
        moving_average_window=3
    )

    result = analyzer.analyze(
        [0.01, 0.02, 0.10, 0.20, 0.30]
    )

    expected = np.mean([0.10, 0.20, 0.30])

    assert result.moving_average == pytest.approx(
        expected
    )


def test_empty_scores_raise_error():
    analyzer = DegradationAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze([])


def test_invalid_scores_raise_error():
    analyzer = DegradationAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analyze(
            [0.1, np.nan, 0.2]
        )

    with pytest.raises(ValueError):
        analyzer.analyze(
            [0.1, np.inf, 0.2]
        )


def test_invalid_thresholds_raise_error():
    with pytest.raises(ValueError):
        DegradationAnalyzer(
            monitor_threshold=0.30,
            early_warning_threshold=0.20,
            high_risk_threshold=0.50,
        )