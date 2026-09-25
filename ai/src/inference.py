from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai.src.anomaly_detection import (
    AnomalyDetector,
    AnomalyPrediction,
)
from ai.src.degradation import (
    DegradationAnalyzer,
    DegradationResult,
)


@dataclass(frozen=True)
class InferenceResult:
    """Combined anomaly and degradation inference result."""

    anomaly: AnomalyPrediction
    degradation: DegradationResult

    @property
    def is_anomalous(self) -> bool:
        """Return whether any anomaly was detected."""

        return self.anomaly.anomaly_count > 0

    @property
    def anomaly_ratio(self) -> float:
        """Return the proportion of anomalous samples."""

        total = len(self.anomaly.labels)

        if total == 0:
            return 0.0

        return self.anomaly.anomaly_count / total


class VibrationInferenceEngine:
    """
    Unified vibration inference engine.

    The engine performs anomaly detection and maintains
    chronological anomaly-score history for degradation analysis.
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
        degradation_analyzer: DegradationAnalyzer | None = None,
    ) -> None:

        if not isinstance(
            anomaly_detector,
            AnomalyDetector,
        ):
            raise TypeError(
                "anomaly_detector must be an "
                "AnomalyDetector instance."
            )

        if degradation_analyzer is not None and not isinstance(
            degradation_analyzer,
            DegradationAnalyzer,
        ):
            raise TypeError(
                "degradation_analyzer must be a "
                "DegradationAnalyzer instance or None."
            )

        self.anomaly_detector = anomaly_detector

        self.degradation_analyzer = (
            degradation_analyzer
            if degradation_analyzer is not None
            else DegradationAnalyzer()
        )

        self._score_history: list[float] = []

    @property
    def score_history(self) -> tuple[float, ...]:
        """Return the accumulated anomaly-score history."""

        return tuple(self._score_history)

    def reset_history(self) -> None:
        """Clear accumulated anomaly-score history."""

        self._score_history.clear()

    def predict(
        self,
        features: pd.DataFrame,
    ) -> InferenceResult:
        """
        Run anomaly detection and degradation analysis.

        Parameters
        ----------
        features:
            Feature dataframe for one or more vibration windows.

        Returns
        -------
        InferenceResult
            Combined anomaly and degradation result.
        """

        anomaly_prediction = self.anomaly_detector.predict(
            features
        )

        self._score_history.extend(
            float(score)
            for score in anomaly_prediction.scores
        )

        degradation_result = (
            self.degradation_analyzer.analyze(
                self._score_history
            )
        )

        return InferenceResult(
            anomaly=anomaly_prediction,
            degradation=degradation_result,
        )