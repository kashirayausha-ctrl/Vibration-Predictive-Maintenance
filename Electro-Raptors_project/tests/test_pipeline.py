import numpy as np
import pytest

from signal_pipeline import PipelineConfig, run_pipeline


def test_pipeline_extracts_features_and_major_frequency() -> None:
    sampling_rate = 100
    time = np.arange(400) / sampling_rate
    data = [
        {
            "timestamp": f"2026-01-01T00:00:{index // 100:02d}.{(index % 100) * 10:03d}",
            "accel_x": value,
            "label": "walking",
        }
        for index, value in enumerate(np.sin(2 * np.pi * 5 * time))
    ]

    result = run_pipeline(
        data,
        ["accel_x"],
        PipelineConfig(sampling_rate=sampling_rate, window_seconds=2, overlap=0.5),
    )

    assert len(result) == 3
    assert "accel_x_rms" in result[0]
    assert all(4.5 <= row["accel_x_dominant_frequency_hz"] <= 5.5 for row in result)
    assert all(row["label"] == "walking" for row in result)


def test_validation_rejects_duplicate_timestamps() -> None:
    data = [
        {"timestamp": "2026-01-01T00:00:00", "accel_x": 1},
        {"timestamp": "2026-01-01T00:00:00", "accel_x": 2},
    ]

    with pytest.raises(ValueError, match="duplicate"):
        run_pipeline(data, ["accel_x"], PipelineConfig(sampling_rate=1))
