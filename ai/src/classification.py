from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationPrediction:
    """Prediction result returned by the fault classifier."""

    labels: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray

    @property
    def confidence(self) -> np.ndarray:
        """Highest class probability for each sample."""
        return np.max(self.probabilities, axis=1)


class FaultClassifier:
    """
    Supervised fault-classification model.

    The final fault labels will be determined from the real
    experimental dataset.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than 0."
            )

        if max_depth is not None and max_depth <= 0:
            raise ValueError(
                "max_depth must be greater than 0 or None."
            )

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        self._feature_names: list[str] | None = None
        self._classes: np.ndarray | None = None
        self._is_fitted = False

    @staticmethod
    def _validate_features(X: pd.DataFrame) -> None:
        """Validate classifier input features."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("X cannot be empty.")

        if X.columns.duplicated().any():
            raise ValueError(
                "X contains duplicate column names."
            )

        non_numeric = X.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        if non_numeric:
            raise TypeError(
                "All classifier features must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

        values = X.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

    @staticmethod
    def _validate_labels(y: pd.Series | np.ndarray) -> None:
        """Validate target labels."""

        if len(y) == 0:
            raise ValueError("y cannot be empty.")

        labels = pd.Series(y)

        if labels.isna().any():
            raise ValueError("y contains missing labels.")

        if labels.nunique() < 2:
            raise ValueError(
                "At least two distinct classes are required."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "FaultClassifier":
        """Train the classifier."""

        self._validate_features(X)
        self._validate_labels(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        self.pipeline.fit(X, y)

        self._feature_names = X.columns.tolist()
        self._classes = self.pipeline.named_steps[
            "classifier"
        ].classes_
        self._is_fitted = True

        return self

    def _validate_schema(self, X: pd.DataFrame) -> None:
        """Ensure prediction features match training features."""

        self._validate_features(X)

        if self._feature_names is None:
            raise RuntimeError(
                "Model feature schema is not available."
            )

        if X.columns.tolist() != self._feature_names:
            raise ValueError(
                "Feature schema mismatch. "
                f"Expected {self._feature_names}, "
                f"received {X.columns.tolist()}."
            )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> ClassificationPrediction:
        """Predict fault classes and probabilities."""

        if not self._is_fitted:
            raise RuntimeError(
                "The classifier must be fitted before prediction."
            )

        self._validate_schema(X)

        labels = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)

        return ClassificationPrediction(
            labels=labels,
            probabilities=probabilities,
            classes=self._classes.copy(),
        )

    def save(self, path: str | Path) -> Path:
        """Save the fitted classifier."""

        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted classifier."
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
                "classes": self._classes,
                "is_fitted": self._is_fitted,
            },
            output_path,
        )

        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "FaultClassifier":
        """Load a previously fitted classifier."""

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        artifact = joblib.load(model_path)

        required_keys = {
            "pipeline",
            "feature_names",
            "classes",
            "is_fitted",
        }

        if not required_keys.issubset(artifact):
            raise ValueError(
                "Invalid fault-classifier artifact."
            )

        classifier = cls()

        classifier.pipeline = artifact["pipeline"]
        classifier._feature_names = artifact["feature_names"]
        classifier._classes = artifact["classes"]
        classifier._is_fitted = artifact["is_fitted"]

        return classifier