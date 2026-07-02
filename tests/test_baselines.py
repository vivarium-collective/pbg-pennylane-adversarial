"""Tests for classical baseline comparison (sklearn LR/RF vs. QML)."""

from __future__ import annotations

import numpy as np
import pytest

from pbg_pennylane_adversarial.baselines import run_baselines


def _linearly_separable_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    split = int(n * 0.7)
    return {
        "train_images": X[:split].tolist(),
        "train_labels": y[:split].tolist(),
        "test_images": X[split:].tolist(),
        "test_labels": y[split:].tolist(),
    }


class TestRunBaselines:
    def test_returns_both_baselines(self):
        out = run_baselines(_linearly_separable_data(), epsilon=0.1)
        assert set(out) == {"logistic_regression", "random_forest"}

    def test_accuracy_keys_and_range(self):
        out = run_baselines(_linearly_separable_data(), epsilon=0.1)
        for metrics in out.values():
            assert set(metrics) == {"benign_accuracy", "adversarial_accuracy"}
            assert 0.0 <= metrics["benign_accuracy"] <= 1.0
            assert 0.0 <= metrics["adversarial_accuracy"] <= 1.0

    def test_easy_problem_high_benign_accuracy(self):
        out = run_baselines(_linearly_separable_data(n=200, seed=1), epsilon=0.05)
        assert out["logistic_regression"]["benign_accuracy"] > 0.8
        assert out["random_forest"]["benign_accuracy"] > 0.8

    def test_zero_epsilon_no_perturbation_drop(self):
        data = _linearly_separable_data()
        out = run_baselines(data, epsilon=0.0)
        for metrics in out.values():
            assert metrics["adversarial_accuracy"] == pytest.approx(
                metrics["benign_accuracy"]
            )

    def test_multiclass(self):
        rng = np.random.default_rng(2)
        n = 90
        X = rng.normal(size=(n, 4))
        y = np.argmax(X[:, :3], axis=1)
        split = 60
        data = {
            "train_images": X[:split].tolist(),
            "train_labels": y[:split].tolist(),
            "test_images": X[split:].tolist(),
            "test_labels": y[split:].tolist(),
        }
        out = run_baselines(data, epsilon=0.05)
        assert set(out) == {"logistic_regression", "random_forest"}


class TestTransferAttack:
    def test_no_transfer_delta_omits_key(self):
        out = run_baselines(_linearly_separable_data(), epsilon=0.1)
        for metrics in out.values():
            assert "transfer_adversarial_accuracy" not in metrics

    def test_empty_transfer_delta_omits_key(self):
        out = run_baselines(_linearly_separable_data(), epsilon=0.1, transfer_delta=[])
        for metrics in out.values():
            assert "transfer_adversarial_accuracy" not in metrics

    def test_transfer_delta_adds_key_in_range(self):
        data = _linearly_separable_data()
        n_test = len(data["test_images"])
        n_features = len(data["test_images"][0])
        delta = np.full((n_test, n_features), 0.1).tolist()
        out = run_baselines(data, epsilon=0.1, transfer_delta=delta)
        for metrics in out.values():
            assert "transfer_adversarial_accuracy" in metrics
            assert 0.0 <= metrics["transfer_adversarial_accuracy"] <= 1.0

    def test_zero_transfer_delta_matches_benign(self):
        data = _linearly_separable_data()
        n_test = len(data["test_images"])
        n_features = len(data["test_images"][0])
        delta = np.zeros((n_test, n_features)).tolist()
        out = run_baselines(data, epsilon=0.1, transfer_delta=delta)
        for metrics in out.values():
            assert metrics["transfer_adversarial_accuracy"] == pytest.approx(
                metrics["benign_accuracy"]
            )
