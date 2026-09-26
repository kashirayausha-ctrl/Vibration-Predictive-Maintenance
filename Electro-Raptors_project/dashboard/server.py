from __future__ import annotations

import csv
import json
import math
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
UI_DIR = Path(__file__).resolve().parent
AI_ROOT = PROJECT_ROOT / "Electro-Raptors" / "Vibration-Predictive-Maintenance"
DATA_FILES = (
    "piezo_predictions.csv",
    "predictions.csv",
    "piezo_features.csv",
    "features.csv",
)


def _source_path() -> Path | None:
    for filename in DATA_FILES:
        path = OUTPUT_DIR / filename
        if path.exists():
            return path
    return None


def _read_source() -> tuple[Path | None, list[dict[str, str]]]:
    path = _source_path()
    if path is None:
        return None, []
    with path.open(newline="", encoding="utf-8") as source:
        return path, list(csv.DictReader(source))


def _number(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _sensor_prefix(rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    for column in rows[0]:
        if column.endswith("_rms"):
            return column.removesuffix("_rms")
    return None


def _condition(label: str | None) -> str:
    normalized = (label or "unknown").strip().lower()
    if normalized in {"idle", "quiet", "healthy", "normal"}:
        return "NORMAL"
    if normalized in {"unknown", ""}:
        return "UNKNOWN"
    return "ATTENTION"


def _validation(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {
            "valid": False,
            "row_count": 0,
            "feature_count": 0,
            "missing_values": 0,
            "non_finite_values": 0,
            "duplicate_windows": 0,
            "accuracy": None,
        }

    columns = list(rows[0])
    numeric_columns = [
        column
        for column in columns
        if column not in {"window_id", "label", "predicted_label", "start_timestamp", "end_timestamp"}
        and all(_number(row.get(column)) is not None for row in rows)
    ]
    missing_values = sum(
        1 for row in rows for column in columns if not row.get(column, "").strip()
    )
    non_finite_values = sum(
        1
        for row in rows
        for column in numeric_columns
        if _number(row.get(column)) is None
    )
    window_ids = [row.get("window_id") for row in rows]
    duplicate_windows = len(window_ids) - len(set(window_ids))
    accuracy = None
    if all("label" in row and "predicted_label" in row for row in rows):
        accuracy = sum(row["label"] == row["predicted_label"] for row in rows) / len(rows)

    return {
        "valid": not missing_values and not non_finite_values and not duplicate_windows,
        "row_count": len(rows),
        "feature_count": len(numeric_columns),
        "missing_values": missing_values,
        "non_finite_values": non_finite_values,
        "duplicate_windows": duplicate_windows,
        "accuracy": accuracy,
    }


def _ai_summary(source: Path | None) -> dict[str, object] | None:
    anomaly_path = AI_ROOT / "ai" / "models" / "anomaly_detector.joblib"
    if source is None or not anomaly_path.exists():
        return None
    try:
        if str(AI_ROOT) not in sys.path:
            sys.path.insert(0, str(AI_ROOT))
        import pandas as pd
        from ai.src.anomaly_detection import AnomalyDetector
        from ai.src.classification import FaultClassifier
        from ai.src.pipeline import run_ai_pipeline

        frame = pd.read_csv(source)
        classifier_path = AI_ROOT / "ai" / "models" / "fault_classifier.joblib"
        classifier = (
            FaultClassifier.load(classifier_path)
            if classifier_path.exists()
            else None
        )
        result = run_ai_pipeline(
            frame,
            AnomalyDetector.load(anomaly_path),
            classifier=classifier,
        )
        return {
            "anomaly_count": result.inference.anomaly.anomaly_count,
            "anomaly_ratio": result.inference.anomaly_ratio,
            "warning_level": result.degradation.warning_level.value,
            "is_degrading": result.degradation.is_degrading,
            "trend_slope": result.degradation.trend_slope,
        }
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        return {"error": str(error)}


def dashboard_state() -> dict[str, object]:
    source, rows = _read_source()
    sensor = _sensor_prefix(rows)
    rms_column = f"{sensor}_rms" if sensor else None
    frequency_column = f"{sensor}_dominant_frequency_hz" if sensor else None
    latest = rows[-1] if rows else {}
    predicted = latest.get("predicted_label") or latest.get("label")

    trend = [
        {
            "window": row.get("window_id"),
            "rms": _number(row.get(rms_column)) if rms_column else None,
            "frequency": _number(row.get(frequency_column)) if frequency_column else None,
        }
        for row in rows[-30:]
    ]
    recent = [
        {
            "window": row.get("window_id"),
            "timestamp": row.get("end_timestamp"),
            "label": row.get("label"),
            "prediction": row.get("predicted_label") or row.get("label"),
            "rms": _number(row.get(rms_column)) if rms_column else None,
            "frequency": _number(row.get(frequency_column)) if frequency_column else None,
        }
        for row in reversed(rows[-8:])
    ]

    return {
        "source": source.name if source else None,
        "sensor": sensor,
        "updated_at": source.stat().st_mtime if source else None,
        "health": {
            "status": _condition(predicted),
            "prediction": predicted or "NO DATA",
            "rms": _number(latest.get(rms_column)) if rms_column else None,
            "frequency": _number(latest.get(frequency_column)) if frequency_column else None,
        },
        "validation": _validation(rows),
        "ai": _ai_summary(source),
        "trend": trend,
        "recent": recent,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/state":
            payload = json.dumps(dashboard_state()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), DashboardHandler)
    print("Dashboard: http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()