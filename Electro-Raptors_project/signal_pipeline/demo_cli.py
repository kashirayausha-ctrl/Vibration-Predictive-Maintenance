from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


def generate_piezo_demo(path: str, sampling_rate: int = 100, seconds_per_label: int = 8) -> None:
    start = datetime(2026, 9, 25, 13, 0, 0)
    rows: list[dict[str, object]] = []
    sample_count = sampling_rate * seconds_per_label * 2
    for index in range(sample_count):
        time_value = index / sampling_rate
        active = index >= sample_count // 2
        label = "tap" if active else "quiet"
        signal_value = (
            2048
            + (650 * math.sin(2 * math.pi * 8 * time_value) if active else 30 * math.sin(2 * math.pi * time_value))
        )
        rows.append({
            "timestamp": (start + timedelta(seconds=time_value)).isoformat(),
            "piezo": round(signal_value, 3),
            "label": label,
        })
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["timestamp", "piezo", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled piezo data for software-only testing.")
    parser.add_argument("output_csv")
    parser.add_argument("--sampling-rate", type=int, default=100)
    parser.add_argument("--seconds-per-label", type=int, default=8)
    args = parser.parse_args()
    generate_piezo_demo(args.output_csv, args.sampling_rate, args.seconds_per_label)
    print(f"Wrote software-only piezo demo data to {args.output_csv}")


if __name__ == "__main__":
    main()
