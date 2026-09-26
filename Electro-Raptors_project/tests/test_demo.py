import csv

from signal_pipeline.demo_cli import generate_piezo_demo


def test_demo_contains_two_piezo_classes(tmp_path) -> None:
    output = tmp_path / "piezo.csv"
    generate_piezo_demo(str(output), sampling_rate=10, seconds_per_label=1)

    with output.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))

    assert len(rows) == 20
    assert {row["label"] for row in rows} == {"quiet", "tap"}
    assert all(row["piezo"] for row in rows)
