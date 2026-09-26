from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .ml import load_feature_csv, load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels from an ML-ready feature CSV.")
    parser.add_argument("features_csv")
    parser.add_argument("model_json")
    parser.add_argument("predictions_csv")
    args = parser.parse_args()

    rows, _ = load_feature_csv(args.features_csv)
    model = load_model(args.model_json)
    predicted_labels = predict(model, rows)
    output_rows = [dict(row, predicted_label=label) for row, label in zip(rows, predicted_labels)]
    output_path = Path(args.predictions_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        fieldnames = list(output_rows[0])
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    labeled_rows = [row for row in output_rows if row.get("label")]
    if labeled_rows:
        correct = sum(row["label"] == row["predicted_label"] for row in labeled_rows)
        print(f"Accuracy: {correct / len(labeled_rows):.2%} ({correct}/{len(labeled_rows)})")
    print(f"Wrote predictions to {args.predictions_csv}")


if __name__ == "__main__":
    main()
