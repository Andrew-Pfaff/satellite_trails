import pytest
import torch

from satellite_trail_segmentation.ml_utils.metrics import (
    best_threshold_by_metric,
    best_threshold_by_penalized_specificity,
    conf_counts_from_arrays,
    conf_counts_from_logits,
    init_conf_counts,
    metrics_from_conf_counts,
    specificity_with_recall_penalty,
    threshold_sweep_from_logits,
    update_conf_counts_batch,
)


def test_confusion_helpers_and_metrics():
    counts = init_conf_counts()
    counts = update_conf_counts_batch(counts, {"tp": 2, "fp": 1, "fn": 1, "tn": 4})
    metrics = metrics_from_conf_counts(counts)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(0.6666666667)


def test_conf_counts_from_logits_and_arrays():
    logits = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    target = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    counts = conf_counts_from_logits(logits, target, threshold=0.5)
    assert counts == {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0}
    assert conf_counts_from_arrays([1, 0, 1, 0], [1, 1, 0, 0]) == counts


def test_threshold_helpers():
    logits = torch.tensor([[3.0], [1.0], [-1.0], [-3.0]])
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    thresholds = [0.2, 0.5, 0.8]
    sweeps = threshold_sweep_from_logits(logits, target, thresholds)
    assert sweeps == {
        0.2: {"tp": 2.0, "fp": 1.0, "fn": 0.0, "tn": 1.0},
        0.5: {"tp": 1.0, "fp": 1.0, "fn": 1.0, "tn": 1.0},
        0.8: {"tp": 1.0, "fp": 0.0, "fn": 1.0, "tn": 2.0},
    }
    metrics = {t: metrics_from_conf_counts(c) for t, c in sweeps.items()}
    best_iou_thr, best_iou_metrics = best_threshold_by_metric(metrics, "iou")
    best_precision_thr, best_precision_metrics = best_threshold_by_metric(metrics, "precision")
    best_recall_thr, best_recall_metrics = best_threshold_by_metric(metrics, "recall")
    best_penalized_thr, best_penalized_metrics = best_threshold_by_penalized_specificity(metrics, min_recall=0.75, penalty=1.0)

    assert best_iou_thr == 0.2
    assert best_iou_metrics["iou"] == pytest.approx(2 / 3)
    assert best_precision_thr == 0.8
    assert best_precision_metrics["precision"] == pytest.approx(1.0)
    assert best_recall_thr == 0.2
    assert best_recall_metrics["recall"] == pytest.approx(1.0)
    assert best_penalized_thr == 0.8
    assert specificity_with_recall_penalty(best_penalized_metrics, 0.75, 1.0) == pytest.approx(0.75)


def test_penalized_specificity():
    metrics = {"specificity": 0.9, "recall": 0.8}
    assert specificity_with_recall_penalty(metrics, 0.85, 2.0) == pytest.approx(0.8)
    best_thr, _ = best_threshold_by_penalized_specificity({0.1: metrics, 0.5: {"specificity": 0.95, "recall": 0.6}}, 0.75, 1.0)
    assert best_thr == 0.1


def test_metric_zero_denominators():
    metrics = metrics_from_conf_counts({"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0})
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["specificity"] == pytest.approx(0.0)
