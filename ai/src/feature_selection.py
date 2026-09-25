from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

if __package__:
    from ai.src.data_validation import (
        DataValidationError,
        validate_or_raise,
    )
else:
    from data_validation import (
        DataValidationError,
        validate_or_raise,
    )


@dataclass(frozen=True)
class FeatureDataset:
    """Validated ML-ready feature dataset."""

    data: pd.DataFrame
    feature_names: tuple[str, ...]


class FeatureInterface:
    """
    Prepare validated feature data for the AI pipeline.

    This interface separates the data-pipeline output from
    the machine-learning models.
    """

    def __init__(
        self,
        required_features: list[str] | tuple[str, ...] | None = None,
        *,
        minimum_rows: int = 10,
    ) -> None:

        if minimum_rows <= 0:
            raise ValueError(
                "minimum_rows must be greater than 0."
            )

        self.required_features = (
            tuple(required_features)
            if required_features is not None
            else None
        )

        self.minimum_rows = minimum_rows

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> FeatureDataset:
        """
        Validate and prepare a feature DataFrame.

        Parameters
        ----------
        df:
            ML-ready feature DataFrame from the data pipeline.

        Returns
        -------
        FeatureDataset
            Validated numeric feature dataset.
        """

        validate_or_raise(
            df,
            required_features=self.required_features,
            minimum_rows=self.minimum_rows,
            allow_missing_values=False,
        )

        non_numeric_columns = [
            column
            for column in df.columns
            if not pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        if non_numeric_columns:
            raise DataValidationError(
                "AI pipeline requires numeric features. "
                f"Non-numeric columns detected: "
                f"{non_numeric_columns}"
            )

        feature_names = tuple(
            str(column)
            for column in df.columns
        )

        prepared_data = df.loc[
            :,
            list(feature_names),
        ].copy()

        return FeatureDataset(
            data=prepared_data,
            feature_names=feature_names,
        )