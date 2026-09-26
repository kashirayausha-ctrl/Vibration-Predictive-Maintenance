import pytest

from signal_pipeline.ml import predict, train_nearest_centroid


def test_nearest_centroid_predicts_two_classes() -> None:
    rows = [
        {"feature": "0", "label": "idle"},
        {"feature": "1", "label": "idle"},
        {"feature": "9", "label": "active"},
        {"feature": "10", "label": "active"},
    ]
    model = train_nearest_centroid(rows, ["feature"])

    assert predict(model, [{"feature": "0.2"}, {"feature": "9.8"}]) == ["idle", "active"]


def test_training_requires_multiple_labels() -> None:
    with pytest.raises(ValueError, match="two distinct labels"):
        train_nearest_centroid([{"feature": "1", "label": "idle"}], ["feature"])
