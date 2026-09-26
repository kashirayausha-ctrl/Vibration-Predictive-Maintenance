from dashboard.server import dashboard_state


def test_dashboard_state_reads_backend_artifacts() -> None:
    state = dashboard_state()

    assert state["source"] in {
        "piezo_predictions.csv",
        "predictions.csv",
        "piezo_features.csv",
        "features.csv",
    }
    assert state["validation"]["row_count"] > 0
    assert state["health"]["prediction"]
    assert state["ai"] is not None
    assert "warning_level" in state["ai"]