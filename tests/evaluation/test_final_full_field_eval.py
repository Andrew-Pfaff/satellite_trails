import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "python" / "evaluate_final_full_field.py"
SPEC = importlib.util.spec_from_file_location("evaluate_final_full_field", SCRIPT_PATH)
evaluate_final_full_field = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_final_full_field)


class ConstantUNet(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.size(0), 1, x.size(2), x.size(3)), 10.0, device=x.device, dtype=x.dtype)


class PositiveClassifier(torch.nn.Module):
    def forward(self, x):
        return torch.full((x.size(0), 1), 10.0, device=x.device, dtype=x.dtype)


def write_png_pair(directory, index, split_id):
    image = np.full((32, 32), index * 20, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    image_name = f"field_{index}.fits_full.png"
    mask_name = f"field_{index}_mask.png"
    Image.fromarray(image).save(directory / image_name)
    Image.fromarray(mask).save(directory / mask_name)
    return [image_name, mask_name, str(split_id)]


def test_test_rows_selects_split_and_resolves_paths(tmp_path):
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    rows = [write_png_pair(png_dir, 0, 0), write_png_pair(png_dir, 1, 2)]
    split_csv = tmp_path / "master_split.csv"
    with split_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    selected = evaluate_final_full_field.test_rows(split_csv, png_dir, split_id="2")

    assert len(selected) == 1
    assert selected[0]["image_name"] == "field_1.fits_full.png"
    assert selected[0]["image_path"] == png_dir / "field_1.fits_full.png"
    assert selected[0]["mask_path"] == png_dir / "field_1_mask.png"


def test_postprocess_methods_are_final_report_configs():
    methods = evaluate_final_full_field.postprocess_methods()

    assert set(methods) == {
        "postprocess_asta",
    }
    assert all(config["contour_filter"] is True for config in methods.values())
    assert all(config["contour_area_threshold"] == 3000 for config in methods.values())


def test_aggregate_rows_uses_summed_counts():
    rows = [
        {"method": "a", "tp": 1, "fp": 1, "fn": 0, "tn": 2},
        {"method": "a", "tp": 3, "fp": 0, "fn": 1, "tn": 1},
    ]

    aggregate = evaluate_final_full_field.aggregate_rows(rows)

    assert len(aggregate) == 1
    assert aggregate[0]["image_count"] == 2
    assert aggregate[0]["tp"] == 4
    assert aggregate[0]["fp"] == 1
    assert aggregate[0]["fn"] == 1
    assert aggregate[0]["tn"] == 3
    assert aggregate[0]["iou"] == pytest.approx(4 / 6)


def test_evaluate_full_fields_writes_expected_csvs(monkeypatch, tmp_path):
    png_dir = tmp_path / "png"
    png_dir.mkdir()
    rows = [write_png_pair(png_dir, 0, 2), write_png_pair(png_dir, 1, 2)]
    split_csv = tmp_path / "master_split.csv"
    with split_csv.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    def fake_build_model(model_class, checkpoint_path, allowed_keys):
        if model_class.__name__ == "UNet":
            return ConstantUNet()
        return PositiveClassifier()

    monkeypatch.setattr(evaluate_final_full_field, "build_model", fake_build_model)
    args = argparse.Namespace(
        png_dir=png_dir,
        master_split_csv=split_csv,
        unet_checkpoint="unused_unet.pt",
        classifier_checkpoint="unused_classifier.pt",
        split_id="2",
        unet_threshold=0.6,
        classifier_threshold=0.67,
        normalization="source_zscore",
        patch_dim=32,
        device="cpu",
        unet_batch_size=1,
        classifier_batch_size=1,
        num_workers=0,
        output_dir=tmp_path / "outputs",
    )

    evaluate_final_full_field.evaluate_full_fields(args)

    expected_methods = {
        "unet",
        "classifier_unet",
        "unet_postprocess_asta",
        "classifier_unet_postprocess_asta",
    }
    for method in expected_methods:
        path = args.output_dir / f"{method}_per_image_metrics.csv"
        assert path.exists()
        with path.open() as file:
            assert len(list(csv.DictReader(file))) == 2

    aggregate_path = args.output_dir / "aggregate_metrics.csv"
    assert aggregate_path.exists()
    with aggregate_path.open() as file:
        aggregate_rows = list(csv.DictReader(file))
    assert {row["method"] for row in aggregate_rows} == expected_methods
