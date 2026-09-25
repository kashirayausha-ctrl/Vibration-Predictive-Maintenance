from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class WarningLevel(str, Enum):
    """Machine degradation warning levels."""

    HEALTHY = "healthy"
    MONITOR = "monitor"
    EARLY_WARNING = "early_warning"
    HIGH_RISK = "high_risk"


@dataclass(frozen=True)
class DegradationResult:
    """Result of degradation analysis."""

    current_score: float
    moving_average: float
    trend_slope: float
    score_change: float
    warning_level: WarningLevel
    is_degrading: bool


class DegradationAnalyzer:
    """
    Analyze anomaly-score history to identify degradation trends.

    This module does not estimate remaining useful life.
    It identifies increasing abnormality and generates
    an early-warning state.
    """

    def __init__(
        self,
        *,
        moving_average_window: int = 5,
        monitor_threshold: float = 0.10,
        early_warning_threshold: float = 0.25,
        high_risk_threshold: float = 0.50,
        trend_threshold: float = 0.01,
    ) -> None:

        if moving_average_window <= 0:
            raise ValueError(
                "moving_average_window must be greater than 0."
            )

        if not (
            0.0
            <= monitor_threshold
            < early_warning_threshold
            < high_risk_threshold
        ):
            raise ValueError(
                "Thresholds must satisfy: "
                "0 <= monitor < early_warning < high_risk."
            )

        if trend_threshold < 0.0:
            raise ValueError(
                "trend_threshold cannot be negative."
            )

        self.moving_average_window = moving_average_window
        self.monitor_threshold = monitor_threshold
        self.early_warning_threshold = early_warning_threshold
        self.high_risk_threshold = high_risk_threshold
        self.trend_threshold = trend_threshold

    @staticmethod
    def _validate_scores(
        scores: np.ndarray | list[float],
    ) -> np.ndarray:
        """Validate and normalize anomaly scores."""

        values = np.asarray(scores, dtype=float)

        if values.ndim != 1:
            raise ValueError(
                "scores must be a one-dimensional sequence."
            )

        if values.size == 0:
            raise ValueError(
                "scores cannot be empty."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "scores contain NaN or infinite values."
            )

        return values

    def _calculate_trend(
        self,
        scores: np.ndarray,
    ) -> float:
        """Calculate the linear trend slope."""

        if len(scores) < 2:
            return 0.0

        x = np.arange(len(scores), dtype=float)

        slope = np.polyfit(
            x,
            scores,
            deg=1,
        )[0]

        return float(slope)

    def _determine_warning_level(
        self,
        current_score: float,
    ) -> WarningLevel:

        if current_score >= self.high_risk_threshold:
            return WarningLevel.HIGH_RISK

        if current_score >= self.early_warning_threshold:
            return WarningLevel.EARLY_WARNING

        if current_score >= self.monitor_threshold:
            return WarningLevel.MONITOR

        return WarningLevel.HEALTHY

    def analyze(
        self,
        scores: np.ndarray | list[float],
    ) -> DegradationResult:
        """
        Analyze an anomaly-score history.

        Parameters
        ----------
        scores:
            Chronological anomaly scores.

        Returns
        -------
        DegradationResult
            Current degradation state and trend.
        """

        values = self._validate_scores(scores)

        window = values[
            -self.moving_average_window :
        ]

        current_score = float(values[-1])

        moving_average = float(
            np.mean(window)
        )

        trend_slope = self._calculate_trend(
            window
        )

        if len(values) >= 2:
            score_change = float(
                values[-1] - values[-2]
            )
        else:
            score_change = 0.0

        warning_level = self._determine_warning_level(
            current_score
        )

        is_degrading = (
            trend_slope >= self.trend_threshold
        )

        return DegradationResult(
            current_score=current_score,
            moving_average=moving_average,
            trend_slope=trend_slope,
            score_change=score_change,
            warning_level=warning_level,
            is_degrading=is_degrading,
        )