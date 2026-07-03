"""Tests for the A1 phase-4 classical dynamics baselines (persistence/DMD/SINDy).

Sanity/recovery tests on small synthetic systems only -- the real held-out-
trajectory run against real WCM data belongs in
docs/investigation_a1_qgrnn_surrogate/run_baseline_gate.py, not pytest
(mirrors qgrnn_surrogate.py's own testing convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from pbg_pennylane_adversarial.dynamics_baselines import (
    fit_dmd,
    fit_sindy,
    per_node_metrics,
    persistence_predict,
)


class TestPersistence:
    def test_predicts_input_unchanged(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.array_equal(persistence_predict(X), X)

    def test_returns_copy_not_view(self):
        X = np.array([[1.0, 2.0]])
        pred = persistence_predict(X)
        pred[0, 0] = 999.0
        assert X[0, 0] == 1.0


class TestFitDMD:
    def test_recovers_known_linear_operator(self):
        rng = np.random.default_rng(0)
        d = 3
        A_true = rng.normal(scale=0.3, size=(d, d))
        X = rng.normal(size=(200, d))
        Y = X @ A_true.T

        result = fit_dmd(X, Y)
        assert np.allclose(result.operator, A_true, atol=1e-6)
        pred = result.predict(X)
        assert np.allclose(pred, Y, atol=1e-6)

    def test_beats_persistence_on_linear_system(self):
        rng = np.random.default_rng(1)
        d = 4
        A_true = rng.normal(scale=0.2, size=(d, d))
        X = rng.normal(size=(150, d))
        Y = X @ A_true.T + rng.normal(scale=0.01, size=(150, d))

        dmd_pred = fit_dmd(X, Y).predict(X)
        persistence_pred = persistence_predict(X)

        dmd_err = np.mean((Y - dmd_pred) ** 2)
        persistence_err = np.mean((Y - persistence_pred) ** 2)
        assert dmd_err < persistence_err

    def test_handles_severely_collinear_features(self):
        # Mirrors the real phase-1 finding: near-duplicate columns should not
        # blow up the pseudoinverse (condition number ~1e14 observed on real
        # mass-group data).
        rng = np.random.default_rng(2)
        base = rng.normal(size=(100, 1))
        X = np.hstack([base, base * 2 + 1e-12, base * 3 + 1e-12])
        Y = X * 1.01

        result = fit_dmd(X, Y)
        assert np.all(np.isfinite(result.operator))
        assert result.rank_used <= 3

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_dmd(np.zeros((5, 2)), np.zeros((5, 3)))

    def test_explicit_rank_is_respected(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 5))
        Y = rng.normal(size=(50, 5))
        result = fit_dmd(X, Y, rank=2)
        assert result.rank_used == 2


class TestFitSINDy:
    def test_recovers_sparse_linear_system(self):
        rng = np.random.default_rng(4)
        d = 3
        X = rng.uniform(-1, 1, size=(300, d))
        # y0 = 2*x0, y1 = -1*x1, y2 = 0 (all sparse, no cross terms).
        Y = np.stack([2.0 * X[:, 0], -1.0 * X[:, 1], np.zeros(300)], axis=1)

        result = fit_sindy(X, Y, degree=2, threshold=0.05)
        pred = result.predict(X)
        assert np.allclose(pred, Y, atol=0.05)

        active0 = dict(result.active_terms(0))
        assert "x0" in active0
        assert active0["x0"] == pytest.approx(2.0, abs=0.05)

    def test_recovers_pairwise_nonlinear_term(self):
        rng = np.random.default_rng(5)
        X = rng.uniform(-1, 1, size=(400, 2))
        Y = (1.5 * X[:, 0] * X[:, 1])[:, None]

        result = fit_sindy(X, Y, degree=2, threshold=0.05)
        active = dict(result.active_terms(0))
        assert "x0*x1" in active
        assert active["x0*x1"] == pytest.approx(1.5, abs=0.05)

    def test_sample_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            fit_sindy(np.zeros((5, 2)), np.zeros((4, 2)))

    def test_predict_shape(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(20, 3))
        Y = rng.normal(size=(20, 3))
        result = fit_sindy(X, Y)
        assert result.predict(X).shape == (20, 3)


class TestPerNodeMetrics:
    def test_perfect_prediction_gives_r2_one(self):
        Y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        metrics = per_node_metrics(Y, Y.copy(), names=["a", "b"])
        assert metrics["per_node"]["a"]["r2"] == pytest.approx(1.0)
        assert metrics["per_node"]["b"]["r2"] == pytest.approx(1.0)
        assert metrics["summary"]["frac_r2_above_0.5"] == pytest.approx(1.0)

    def test_mean_prediction_gives_r2_zero(self):
        Y = np.array([[1.0], [2.0], [3.0]])
        pred = np.full_like(Y, Y.mean())
        metrics = per_node_metrics(Y, pred)
        assert metrics["per_node"]["x0"]["r2"] == pytest.approx(0.0, abs=1e-9)

    def test_zero_variance_column_excluded_from_summary(self):
        Y = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        pred = np.array([[1.0, 5.5], [1.0, 6.0], [1.0, 6.5]])
        metrics = per_node_metrics(Y, pred, names=["const", "varies"])
        assert np.isnan(metrics["per_node"]["const"]["r2"])
        assert not np.isnan(metrics["per_node"]["varies"]["r2"])
