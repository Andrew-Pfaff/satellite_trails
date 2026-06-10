import csv

from satellite_trail_segmentation.utils.h5summary import build_h5_summary, write_summary_csv


def test_build_h5_summary_and_write_csv(dummy_h5_file, tmp_path):
    rows = build_h5_summary(dummy_h5_file)
    assert len(rows) == 3
    assert rows[0]["split"] == "train"
    assert rows[1]["split"] == "val"
    assert rows[2]["split"] == "test"

    csv_path = tmp_path / "summary.csv"
    write_summary_csv(rows, csv_path)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        out = list(reader)
    assert len(out) == 3
    assert out[0]["source_file"] == "source_0.png"
