from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

DataRow = Mapping[str, object]
FeatureRow = dict[str, object]


@dataclass(frozen=True)
class PipelineConfig:
    sampling_rate: float
    window_seconds: float = 2.0
    overlap: float = 0.5
    lowcut_hz: float | None = None
    highcut_hz: float | None = None
    filter_order: int = 4

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in the range [0, 1)")
        if self.lowcut_hz is None and self.highcut_hz is None:
            return
        nyquist = self.sampling_rate / 2
        if self.lowcut_hz is not None and self.lowcut_hz <= 0:
            raise ValueError("lowcut_hz must be positive")
        if self.highcut_hz is not None and self.highcut_hz <= 0:
            raise ValueError("highcut_hz must be positive")
        if self.highcut_hz is not None and self.highcut_hz >= nyquist:
            raise ValueError("highcut_hz must be below the Nyquist frequency")
        if self.lowcut_hz is not None and self.highcut_hz is not None and self.lowcut_hz >= self.highcut_hz:
            raise ValueError("lowcut_hz must be lower than highcut_hz")


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid timestamp value: {value!r}") from error


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def validate_data(data: Sequence[DataRow], sensor_columns: Sequence[str]) -> None:
    if not data:
        raise ValueError("Input data is empty")
    missing = {"timestamp", *sensor_columns}.difference(data[0].keys())
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    timestamps = [_parse_timestamp(row["timestamp"]) for row in data]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("timestamp contains duplicate values")
    for column in sensor_columns:
        numeric_count = sum(np.isfinite(_to_float(row[column])) for row in data)
        if numeric_count == 0:
            raise ValueError(f"Sensor column {column!r} contains no numeric values")


def _interpolate(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    if not valid.any():
        raise ValueError("A sensor column contains no numeric values")
    positions = np.arange(len(array))
    return np.interp(positions, positions[valid], array[valid])


def _prepare_data(data: Sequence[DataRow], sensor_columns: Sequence[str]) -> tuple[list[datetime], dict[str, np.ndarray], list[object] | None]:
    ordered = sorted(data, key=lambda row: _parse_timestamp(row["timestamp"]))
    timestamps = [_parse_timestamp(row["timestamp"]) for row in ordered]
    values = {
        column: _interpolate([_to_float(row[column]) for row in ordered])
        for column in sensor_columns
    }
    labels = [row.get("label") for row in ordered] if "label" in ordered[0] else None
    return timestamps, values, labels


def _filter_signal(values: np.ndarray, config: PipelineConfig) -> np.ndarray:
    if config.lowcut_hz is None and config.highcut_hz is None:
        return values
    def low_pass(cutoff_hz: float, source: np.ndarray) -> np.ndarray:
        kernel_size = max(1, round(config.sampling_rate / cutoff_hz))
        kernel = np.ones(kernel_size) / kernel_size
        padded = np.pad(source, (kernel_size // 2,), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(source)]

    if config.lowcut_hz is not None:
        high_pass = values - low_pass(config.lowcut_hz, values)
    else:
        high_pass = values
    if config.highcut_hz is not None:
        return low_pass(config.highcut_hz, high_pass)
    return high_pass


def _window_features(values: np.ndarray, sampling_rate: float) -> dict[str, float]:
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))
    frequencies = np.fft.rfftfreq(len(centered), d=1 / sampling_rate)
    dominant_index = int(np.argmax(spectrum[1:]) + 1) if len(spectrum) > 1 else 0
    power = spectrum**2
    power_total = float(np.sum(power))
    probability = power / power_total if power_total else np.zeros_like(power)
    nonzero_probability = probability[probability > 0]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "peak_to_peak": float(np.ptp(values)),
        "dominant_frequency_hz": float(frequencies[dominant_index]),
        "spectral_energy": power_total,
        "spectral_entropy": float(-np.sum(nonzero_probability * np.log2(nonzero_probability))),
    }


def run_pipeline(data: Sequence[DataRow], sensor_columns: Sequence[str], config: PipelineConfig) -> list[FeatureRow]:
    """Validate and transform sensor data into one feature row per signal window."""
    validate_data(data, sensor_columns)
    timestamps, raw_values, labels = _prepare_data(data, sensor_columns)
    window_size = round(config.window_seconds * config.sampling_rate)
    step = round(window_size * (1 - config.overlap))
    if window_size < 2 or step < 1:
        raise ValueError("window_seconds and overlap produce an invalid window")

    filtered = {column: _filter_signal(values, config) for column, values in raw_values.items()}
    rows: list[FeatureRow] = []
    for window_id, start in enumerate(range(0, len(timestamps) - window_size + 1, step)):
        row: FeatureRow = {
            "window_id": window_id,
            "start_timestamp": timestamps[start].isoformat(),
            "end_timestamp": timestamps[start + window_size - 1].isoformat(),
        }
        for column in sensor_columns:
            features = _window_features(filtered[column][start : start + window_size], config.sampling_rate)
            row.update({f"{column}_{name}": value for name, value in features.items()})
        if labels is not None:
            window_labels = labels[start : start + window_size]
            row["label"] = max(set(window_labels), key=window_labels.count)
        rows.append(row)
    return rows


def run_csv(input_path: str, output_path: str, sensor_columns: Sequence[str], config: PipelineConfig) -> list[FeatureRow]:
    with Path(input_path).open(newline="", encoding="utf-8") as input_file:
        data = list(csv.DictReader(input_file))
    features = run_pipeline(data, sensor_columns, config)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(features[0]) if features else ["window_id"]
    with Path(output_path).open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
    return features
