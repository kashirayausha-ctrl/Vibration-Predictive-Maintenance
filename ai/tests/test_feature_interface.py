import numpy as np
import pandas as pd
import pytest

from ai.src.data_validation import DataValidationError
from ai.src.feature_interface import FeatureDataset, FeatureInterface


FEATURES = [
    "RMS",
    "Peak",
    "Std",
    "Kurtosis",
    "Dominant_Frequency",
]


def make_feature_data(rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        {
            "RMS": rng.normal(0.20, 0.02, rows),
            "Peak": rng.normal(0.45, 0.04, rows),
            "Std": rng.normal(0.10, 0.01, rows),
            "Kurtosis": rng.normal(3.0, 0.20, rows),
            "Dominant_Frequency": rng.normal(150.0, 3.0, rows),
        }
    )


def test_prepare_returns_feature_dataset() -> None:
    df = make_feature_data()

    interface = FeatureInterface(
        required_features=FEATURES,
    )

    result = interface.prepare(df)

    assert isinstance(result, FeatureDataset)


def test_prepare_preserves_feature_names() -> None:
    df = make_feature_data()

    interface = FeatureInterface(
        required_features=FEATURES,
    )

    result = interface.prepare(df)

    assert result.feature_names == tuple(FEATURES)


def test_prepare_preserves_data_shape() -> None:
    df = make_feature_data(rows=25)

    interface = FeatureInterface(
        required_features=FEATURES,
    )

    result = interface.prepare(df)

    assert result.data.shape == (25, 5)


def test_prepare_returns_copy() -> None:
    df = make_feature_data()

    interface = FeatureInterface(
        required_features=FEATURES,
    )

    result = interface.prepare(df)

    assert result.data is not df


def test_missing_required_feature_raises() -> None:
    df = make_feature_data().drop(
        columns=["RMS"]
    )

    interface = FeatureInterface(
        required_features=FEATURES,
    )

    with pytest.raises(DataValidationError):
        interface.prepare(df)


def test_non_numeric_feature_raises() -> None:
    df = make_feature_data()

    df["Sensor_ID"] = "ESP32"

    interface = FeatureInterface()

    with pytest.raises(DataValidationError):
        interface.prepare(df)


def test_missing_values_raise() -> None:
    df = make_feature_data()

    df.loc[0, "RMS"] = np.nan

    interface = FeatureInterface()

    with pytest.raises(DataValidationError):
        interface.prepare(df)


def test_infinite_values_raise() -> None:
    df = make_feature_data()

    df.loc[0, "RMS"] = np.inf

    interface = FeatureInterface()

    with pytest.raises(DataValidationError):
        interface.prepare(df)


def test_insufficient_rows_raise() -> None:
    df = make_feature_data(rows=5)

    interface = FeatureInterface(
        minimum_rows=10,
    )

    with pytest.raises(DataValidationError):
        interface.prepare(df)


def test_invalid_minimum_rows_raise() -> None:
    with pytest.raises(ValueError):
        FeatureInterface(
            minimum_rows=0,
        )