import numpy as np

from src.metrics import best_threshold_by_f1, compute_metrics


def test_compute_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_compute_metrics_all_wrong():
    y_true = np.array([0, 1])
    y_prob = np.array([0.9, 0.1])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_best_threshold_by_f1_prefers_separating_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.2, 0.8, 0.85, 0.95])
    t = best_threshold_by_f1(y_true, y_prob)
    assert 0.2 < t < 0.8
