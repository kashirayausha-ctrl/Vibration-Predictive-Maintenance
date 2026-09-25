import pandas as pd
import pytest

from ai.src.data_validation import (
    DataValidationError,
    validate_feature_dataframe,
    validate_or_raise,
)


def test_valid_feature_dataframe() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, 0.30, 0.25],
            "Peak": [0.40, 0.60, 0.50],
        }
    )

    result = validate_feature_dataframe(
        df,
        minimum_rows=3,
    )

    assert result.valid is True
    assert result.errors == ()
    assert result.row_count == 3
    assert result.feature_count == 2


def test_empty_dataframe_is_invalid() -> None:
    df = pd.DataFrame()

    result = validate_feature_dataframe(df)

    assert result.valid is False
    assert any(
        "no rows" in error.lower()
        for error in result.errors
    )


def test_missing_required_feature_is_invalid() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, 0.30, 0.25],
        }
    )

    result = validate_feature_dataframe(
        df,
        required_features=["RMS", "Peak"],
        minimum_rows=3,
    )

    assert result.valid is False
    assert any(
        "missing required feature" in error.lower()
        for error in result.errors
    )


def test_nan_values_are_rejected() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, None, 0.25],
            "Peak": [0.40, 0.60, 0.50],
        }
    )

    result = validate_feature_dataframe(
        df,
        minimum_rows=3,
    )

    assert result.valid is False
    assert any(
        "missing values" in error.lower()
        for error in result.errors
    )


def test_infinite_values_are_rejected() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, float("inf"), 0.25],
            "Peak": [0.40, 0.60, 0.50],
        }
    )

    result = validate_feature_dataframe(
        df,
        minimum_rows=3,
    )

    assert result.valid is False
    assert any(
        "non-finite" in error.lower()
        for error in result.errors
    )


def test_constant_feature_generates_warning() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, 0.20, 0.20],
            "Peak": [0.40, 0.60, 0.50],
        }
    )

    result = validate_feature_dataframe(
        df,
        minimum_rows=3,
    )

    assert result.valid is True
    assert any(
        "constant feature" in warning.lower()
        for warning in result.warnings
    )


def test_validate_or_raise_raises_on_invalid_data() -> None:
    df = pd.DataFrame(
        {
            "RMS": [0.20, None, 0.25],
            "Peak": [0.40, 0.60, 0.50],
        }
    )

    with pytest.raises(DataValidationError):
        validate_or_raise(
            df,
            minimum_rows=3,
        )