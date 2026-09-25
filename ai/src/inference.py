from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai.src.anomaly_detection import AnomalyDetector, AnomalyPrediction


@dataclass(frozen=True)
class InferenceResult:
    """Unified result returned by the AI inference layer."""

    anomaly: AnomalyPrediction

    @property
    def is_anomalous(self) -> bool:
        """Return True if at least one sample is anomalous."""
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
    Unified inference interface for the vibration AI system.

    Additional models such as fault classification and degradation
    prediction will be integrated here later.
    """

    def __init__(
        self,
        anomaly_detector: AnomalyDetector,
    ) -> None:
        if not isinstance(anomaly_detector, AnomalyDetector):
            raise TypeError(
                "anomaly_detector must be an AnomalyDetector instance."
            )

        self.anomaly_detector = anomaly_detector

    def predict(
        self,
        features: pd.DataFrame,
    ) -> InferenceResult:
        """
        Run the complete currently available inference pipeline.
        """

        anomaly_prediction = self.anomaly_detector.predict(features)

        return InferenceResult(
            anomaly=anomaly_prediction,
        )