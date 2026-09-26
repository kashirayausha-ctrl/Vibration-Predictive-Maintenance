from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

_METADATA_COLUMNS = {"window_id", "start_timestamp", "end_timestamp", "label"}


def load_feature_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError("Feature CSV is empty")
    feature_names = [
        name for name in rows[0]
        if name not in _METADATA_COLUMNS and all(_is_number(row[name]) for row in rows)
    ]
    if not feature_names:
        raise ValueError("Feature CSV contains no numeric feature columns")
    return rows, feature_names


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _matrix(rows: Sequence[dict[str, str]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def train_nearest_centroid(rows: Sequence[dict[str, str]], feature_names: Sequence[str]) -> dict[str, object]:
    labels = sorted({row.get("label", "") for row in rows})
    if len(labels) < 2:
        raise ValueError("Training requires at least two distinct labels")
    values = _matrix(rows, feature_names)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (values - mean) / scale
    centroids = {
        label: normalized[[row.get("label") == label for row in rows]].mean(axis=0).tolist()
        for label in labels
    }
    return {
        "algorithm": "nearest_centroid",
        "feature_names": list(feature_names),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "centroids": centroids,
    }


def predict(model: dict[str, object], rows: Sequence[dict[str, str]]) -> list[str]:
    feature_names = model["feature_names"]
    values = _matrix(rows, feature_names)
    normalized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    centroids = {
        label: np.asarray(vector, dtype=float)
        for label, vector in model["centroids"].items()
    }
    return [
        min(centroids, key=lambda label: float(np.linalg.norm(row - centroids[label])))
        for row in normalized
    ]


def save_model(model: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
