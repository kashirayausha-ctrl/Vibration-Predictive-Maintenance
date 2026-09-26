from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from .ml import load_model, predict
from .pipeline import PipelineConfig, run_pipeline


def _read_sensor_row(line: str, sensor_columns: list[str]) -> dict[str, object] | None:
    values = next(csv.reader([line.strip()]), [])
    if not values or values[0].lower() in {"timestamp", "piezo"}:
        return None
    if len(values) == len(sensor_columns) + 1:
        timestamp, *sensor_values = values
    elif len(values) == len(sensor_columns):
        timestamp = datetime.now(timezone.utc).isoformat()
        sensor_values = values
    else:
        raise ValueError(
            f"Expected {len(sensor_columns)} sensor values or timestamp plus values, got {len(values)}"
        )
    row: dict[str, object] = {"timestamp": timestamp}
    row.update(dict(zip(sensor_columns, sensor_values)))
    return row


def capture_serial(
    port: str,
    baudrate: int,
    sensor_columns: list[str],
    duration_seconds: float,
    config: PipelineConfig,
    model_path: str | None = None,
    raw_output: str | None = None,
) -> list[dict[str, object]]:
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as error:
        raise RuntimeError("Install serial support with: py -m pip install pyserial") from error

    rows: list[dict[str, object]] = []
    deadline = time.monotonic() + duration_seconds
    try:
        connection = serial.Serial(port, baudrate, timeout=1)
    except serial.SerialException as error:
        detected_ports = [device.device for device in list_ports.comports()]
        available = ", ".join(detected_ports) if detected_ports else "none"
        raise RuntimeError(
            f"Cannot open {port}. Detected serial ports: {available}. "
            "Connect the ESP32 with a data cable and use the COM port shown in Device Manager."
        ) from error

    with connection:
        while time.monotonic() < deadline:
            line = connection.readline().decode("utf-8", errors="ignore")
            if not line.strip():
                continue
            row = _read_sensor_row(line, sensor_columns)
            if row is not None:
                rows.append(row)
    if raw_output:
        _write_rows(raw_output, rows, sensor_columns)
    features = run_pipeline(rows, sensor_columns, config)
    if model_path and features:
        model = load_model(model_path)
        labels = predict(model, features)
        for feature_row, label in zip(features, labels):
            print(json.dumps({"window_id": feature_row["window_id"], "prediction": label}))
    return features


def _write_rows(path: str, rows: list[dict[str, object]], sensor_columns: list[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["timestamp", *sensor_columns])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read sensor values directly from an ESP32 serial stream.")
    parser.add_argument("--port", required=True, help="Serial port, for example COM5")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--sensors", nargs="+", required=True)
    parser.add_argument("--duration-seconds", type=float, default=10)
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--window-seconds", type=float, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--model")
    parser.add_argument("--raw-output", help="Optional audit CSV; capture remains automatic")
    args = parser.parse_args()
    try:
        features = capture_serial(
            args.port,
            args.baudrate,
            args.sensors,
            args.duration_seconds,
            PipelineConfig(args.sampling_rate, args.window_seconds, args.overlap),
            args.model,
            args.raw_output,
        )
    except RuntimeError as error:
        parser.exit(1, f"serial-pipeline: {error}\n")
    print(f"Processed {len(features)} feature windows from {args.port}")


if __name__ == "__main__":
    main()
