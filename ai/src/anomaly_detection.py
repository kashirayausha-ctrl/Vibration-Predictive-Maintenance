from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AnomalyPrediction:
    """Prediction result for a batch of feature vectors."""

    labels: np.ndarray
    scores: np.ndarray

    @property
    def anomaly_count(self) -> int:
        """Number of samples classified as anomalous."""
        return int(np.sum(self.labels == -1))

    @property
    def normal_count(self) -> int:
        """Number of samples classified as normal."""
        return int(np.sum(self.labels == 1))


class AnomalyDetector:
    """
    Isolation Forest based anomaly detector.

    The detector is intended to be trained primarily on healthy
    vibration feature data.
    """

    def __init__(
        self,
        *,
        contamination: float = "auto",
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        if isinstance(contamination, float):
            if not 0.0 < contamination <= 0.5:
                raise ValueError(
                    "contamination must be between 0 and 0.5."
                )

        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0.")

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "detector",
                    IsolationForest(
                        n_estimators=n_estimators,
                        contamination=contamination,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self._feature_names: list[str] | None = None
        self._is_fitted = False

    @staticmethod
    def _validate_input(
        X: pd.DataFrame,
    ) -> None:
        """Validate model input."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("X cannot be empty.")

        if X.columns.duplicated().any():
            raise ValueError("X contains duplicate column names.")

        non_numeric = X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise TypeError(
                f"All model features must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

        values = X.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit the anomaly detector using healthy feature data.
        """

        self._validate_input(X)

        self.pipeline.fit(X)

        self._feature_names = X.columns.tolist()
        self._is_fitted = True

        return self

    def _validate_feature_schema(
        self,
        X: pd.DataFrame,
    ) -> None:
        """Ensure prediction features match training features."""

        self._validate_input(X)

        if self._feature_names is None:
            raise RuntimeError("Model feature schema is not available.")

        if X.columns.tolist() != self._feature_names:
            raise ValueError(
                "Feature schema mismatch. "
                f"Expected {self._feature_names}, "
                f"received {X.columns.tolist()}."
            )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> AnomalyPrediction:
        """
        Predict normal/anomalous samples.

        Returns
        -------
        AnomalyPrediction
            labels:
                1  = normal
                -1 = anomalous

            scores:
                Higher values indicate greater anomaly.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "The anomaly detector must be fitted before prediction."
            )

        self._validate_feature_schema(X)

        labels = self.pipeline.predict(X)

        # Isolation Forest's decision_function gives higher values
        # to normal observations. Negating it makes higher scores
        # correspond to stronger anomalies.
        scores = -self.pipeline.decision_function(X)

        return AnomalyPrediction(
            labels=labels,
            scores=scores,
        )

    def save(self, path: str | Path) -> Path:
        """Save the fitted detector to disk."""

        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted anomaly detector."
            )

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self._feature_names,
                "is_fitted": self._is_fitted,
            },
            output_path,
        )

        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "AnomalyDetector":
        """Load a previously fitted detector."""

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        artifact = joblib.load(model_path)

        required_keys = {
            "pipeline",
            "feature_names",
            "is_fitted",
        }

        if not required_keys.issubset(artifact):
            raise ValueError(
                "Invalid anomaly detector artifact."
            )

        detector = cls()
        detector.pipeline = artifact["pipeline"]
        detector._feature_names = artifact["feature_names"]
        detector._is_fitted = artifact["is_fitted"]

        return detector