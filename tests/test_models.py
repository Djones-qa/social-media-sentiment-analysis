"""Unit tests for model evaluation utilities."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.evaluate import compute_metrics


@pytest.fixture
def sample_predictions():
    y_true = np.array(["Positive","Negative","Neutral","Positive","Negative",
                       "Neutral","Positive","Positive","Negative","Neutral"])
    y_pred = np.array(["Positive","Negative","Neutral","Positive","Neutral",
                       "Neutral","Negative","Positive","Negative","Neutral"])
    return y_true, y_pred


class TestComputeMetrics:
    def test_returns_all_keys(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_metrics(y_true, y_pred)
        expected = {"accuracy","f1_weighted","f1_macro","precision_weighted","recall_weighted","n_samples"}
        assert set(metrics.keys()) == expected

    def test_accuracy_range(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert 0 <= metrics["accuracy"] <= 1

    def test_perfect_prediction(self):
        labels = ["Positive","Negative","Neutral"] * 5
        metrics = compute_metrics(labels, labels)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_sample_count(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["n_samples"] == 10

    def test_f1_less_than_or_equal_accuracy(self, sample_predictions):
        y_true, y_pred = sample_predictions
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1_macro"] <= metrics["accuracy"] + 0.01
