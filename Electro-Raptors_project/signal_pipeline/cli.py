from __future__ import annotations

import argparse

from .pipeline import PipelineConfig, run_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sensor CSV data into ML-ready window features.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--sensors", nargs="+", required=True, help="Sensor column names")
    parser.add_argument("--sampling-rate", type=float, required=True, help="Samples per second")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--lowcut-hz", type=float)
    parser.add_argument("--highcut-hz", type=float)
    args = parser.parse_args()
    config = PipelineConfig(
        sampling_rate=args.sampling_rate,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        lowcut_hz=args.lowcut_hz,
        highcut_hz=args.highcut_hz,
    )
    features = run_csv(args.input_csv, args.output_csv, args.sensors, config)
    print(f"Wrote {len(features)} feature windows to {args.output_csv}")


if __name__ == "__main__":
    main()
