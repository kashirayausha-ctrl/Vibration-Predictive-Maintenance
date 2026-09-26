from signal_pipeline.serial_cli import _read_sensor_row


def test_serial_row_with_timestamp() -> None:
    row = _read_sensor_row("2026-09-25T10:00:00,1.0,2.0,9.8", ["accel_x", "accel_y", "accel_z"])

    assert row == {
        "timestamp": "2026-09-25T10:00:00",
        "accel_x": "1.0",
        "accel_y": "2.0",
        "accel_z": "9.8",
    }


def test_serial_row_without_timestamp_gets_one() -> None:
    row = _read_sensor_row("1.0,2.0,9.8", ["accel_x", "accel_y", "accel_z"])

    assert row is not None
    assert row["accel_x"] == "1.0"
    assert "timestamp" in row


def test_serial_ignores_firmware_sensor_banner() -> None:
    assert _read_sensor_row("piezo", ["piezo"]) is None
