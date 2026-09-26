from signal_pipeline.ml import predict


def test_prediction_matches_training_classes() -> None:
    model = {
        "feature_names": ["feature"],
        "mean": [5.0],
        "scale": [5.0],
        "centroids": {"idle": [-0.8], "active": [0.8]},
    }

    assert predict(model, [{"feature": "1"}, {"feature": "9"}]) == ["idle", "active"]
