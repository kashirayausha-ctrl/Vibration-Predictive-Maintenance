from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    """Result returned after validating a feature dataset."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    row_count: int
    feature_count: int


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


def validate_feature_dataframe(
    df: pd.DataFrame,
    required_features: Iterable[str] | None = None,
    *,
    minimum_rows: int = 10,
    allow_missing_values: bool = False,
) -> ValidationResult:
    """
    Validate a feature DataFrame before it enters the ML pipeline.

    Parameters
    ----------
    df:
        Feature DataFrame to validate.

    required_features:
        Feature names that must exist in the DataFrame.

    minimum_rows:
        Minimum number of rows required.

    allow_missing_values:
        Whether NaN values are allowed.

    Returns
    -------
    ValidationResult
        Structured validation result.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(df, pd.DataFrame):
        return ValidationResult(
            valid=False,
            errors=("Input must be a pandas DataFrame.",),
            warnings=(),
            row_count=0,
            feature_count=0,
        )

    row_count, column_count = df.shape

    if row_count == 0:
        errors.append("Dataset contains no rows.")

    if column_count == 0:
        errors.append("Dataset contains no columns.")

    if row_count < minimum_rows:
        errors.append(
            f"Dataset contains {row_count} rows; "
            f"at least {minimum_rows} rows are required."
        )

    if df.columns.duplicated().any():
        duplicated_columns = (
            df.columns[df.columns.duplicated()]
            .astype(str)
            .tolist()
        )

        errors.append(
            f"Duplicate column names detected: {duplicated_columns}"
        )

    if required_features is not None:
        required = list(required_features)
        missing = [column for column in required if column not in df.columns]

        if missing:
            errors.append(
                f"Missing required feature columns: {missing}"
            )

    if not allow_missing_values and df.isna().any().any():
        missing_counts = (
            df.isna()
            .sum()
            .loc[lambda series: series > 0]
            .to_dict()
        )

        errors.append(
            f"Missing values detected: {missing_counts}"
        )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) == 0:
        errors.append("No numeric feature columns were found.")

    if len(numeric_columns) > 0:
        numeric_values = df[numeric_columns].to_numpy(dtype=float)

        if not np.isfinite(numeric_values).all():
            errors.append(
                "Non-finite numeric values detected "
                "(NaN or infinite values)."
            )

    non_numeric_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
    ]

    if non_numeric_columns:
        warnings.append(
            f"Non-numeric columns detected: {non_numeric_columns}"
        )

    constant_columns = [
        column
        for column in numeric_columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        warnings.append(
            f"Constant feature columns detected: {constant_columns}"
        )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        row_count=row_count,
        feature_count=column_count,
    )


def validate_or_raise(
    df: pd.DataFrame,
    required_features: Iterable[str] | None = None,
    *,
    minimum_rows: int = 10,
    allow_missing_values: bool = False,
) -> ValidationResult:
    """
    Validate a feature DataFrame and raise an exception if invalid.
    """

    result = validate_feature_dataframe(
        df,
        required_features=required_features,
        minimum_rows=minimum_rows,
        allow_missing_values=allow_missing_values,
    )

    if not result.valid:
        message = "Dataset validation failed:\n- " + "\n- ".join(
            result.errors
        )
        raise DataValidationError(message)

    return result