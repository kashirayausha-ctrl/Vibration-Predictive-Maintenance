from __future__ import annotations

import argparse

from .ml import load_feature_csv, save_model, train_nearest_centroid


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy nearest-centroid model from feature CSV data.")
    parser.add_argument("features_csv")
    parser.add_argument("model_json")
    args = parser.parse_args()
    rows, feature_names = load_feature_csv(args.features_csv)
    model = train_nearest_centroid(rows, feature_names)
    save_model(model, args.model_json)
    print(f"Trained {model['algorithm']} model with {len(rows)} windows and {len(feature_names)} features")
    print(f"Labels: {', '.join(model['centroids'])}")
    print(f"Wrote model to {args.model_json}")


if __name__ == "__main__":
    main()
