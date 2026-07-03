"""Classical dynamics baselines for A1's phase-4 decision gate.

Three arms, cheapest/most-interpretable first, meant to be cleared *before*
any multi-session QGRNN/GNN training is justified (see `NEXT_STEPS.md` gap 4,
`todo.md` §2 A1 phase 4):

    persistence (no dynamics) -> DMD (linear dynamics) -> SINDy (sparse
    nonlinear dynamics) -> [MLP / GNN / QGRNN, not built here]

All three operate on the `(X, Y, traj_id)` transition-pair contract produced
by `dataset_transform.wcm_loader.build_transition_pairs()` (and, for a
positive control, `dataset_transform.synthetic_dynamics.SyntheticTransitionGenerator`).

DMD is hand-rolled (rank-truncated SVD pseudoinverse) rather than depending on
`pydmd`, per the phase-1 finding that real WCM transition features are
severely collinear (condition number ~1e14, effective rank ~6/8 on the 8-node
mass/chromosome set) -- `pydmd`'s API also expects one sequential snapshot
matrix, not pre-paired multi-trajectory `(X, Y)` pairs, which this module's
inputs already are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Persistence: Y_hat = X (predict no change)
# ---------------------------------------------------------------------------


def persistence_predict(X: np.ndarray) -> np.ndarray:
    """The trivial floor every other arm must clear: predict no change."""
    return X.copy()


# ---------------------------------------------------------------------------
# DMD: linear operator A s.t. Y ~= A @ X, via rank-truncated SVD pseudoinverse
# ---------------------------------------------------------------------------


@dataclass
class DMDResult:
    operator: np.ndarray          # (d, d) fitted linear operator A
    eigenvalues: np.ndarray       # eigenvalues of A (complex, len d)
    singular_values: np.ndarray   # singular values of the training X (all of them)
    rank_used: int                # how many singular values/modes were kept

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.operator.T


def fit_dmd(X: np.ndarray, Y: np.ndarray, rank: int | None = None,
            energy_threshold: float = 1 - 1e-10, ridge: float = 0.0) -> DMDResult:
    """Fit `A` minimizing `||Y - A @ X.T||` via truncated-SVD (Tikhonov-regularized) pseudoinverse.

    `X`, `Y` are `(n_samples, n_features)` transition pairs (columns are
    features, rows are samples) -- the operator solved for satisfies
    `Y[i] ~= A @ X[i]` for each row `i`, i.e. `A = Y.T @ pinv(X.T)`, computed
    via `X.T`'s SVD truncated to `rank` singular values (or, if `rank` is
    None, the smallest rank capturing `energy_threshold` of the singular-value
    energy -- this is the numerical-stability guard: severely collinear real
    WCM features produce a near-singular `X.T` whose small singular values
    would otherwise blow up a naive pseudoinverse).

    `ridge > 0` additionally shrinks the pseudoinverse (`S / (S**2 + ridge)`
    instead of `1/S`) -- needed because an unconstrained `d x d` dense
    operator (`d**2` free parameters) is easy to overfit when the number of
    training transitions is only a few multiples of `d`, which is the actual
    real-WCM regime at N=4 trajectories (confirmed empirically: even a
    perfectly linear synthetic system with only ~36 training transitions
    fits its 8x8 operator to near-zero in-sample error but generalizes badly
    with `ridge=0`). Callers on small-N data should cross-validate `ridge`
    rather than assume 0 is safe.
    """
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} vs {Y.shape}")
    n = X.shape[0]
    if n == 0:
        raise ValueError("cannot fit DMD on zero samples")

    # SVD of the snapshot matrix (features x samples).
    U, S, Vh = np.linalg.svd(X.T, full_matrices=False)

    if rank is None:
        if S[0] <= 0:
            r = 1
        else:
            energy = np.cumsum(S ** 2) / np.sum(S ** 2)
            r = int(np.searchsorted(energy, energy_threshold) + 1)
        r = max(1, min(r, len(S)))
    else:
        r = max(1, min(rank, len(S)))

    U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
    # (regularized) pinv(X.T) truncated to rank r: V_r @ diag(S_r/(S_r^2+ridge)) @ U_r.T
    S_inv_r = S_r / (S_r ** 2 + ridge) if ridge > 0 else 1.0 / S_r
    X_pinv_r = Vh_r.T @ np.diag(S_inv_r) @ U_r.T
    A = Y.T @ X_pinv_r  # (d, d)

    eigenvalues = np.linalg.eigvals(A)
    return DMDResult(operator=A, eigenvalues=eigenvalues, singular_values=S, rank_used=r)


# ---------------------------------------------------------------------------
# SINDy: sparse regression over a small polynomial candidate-term library
# (Brunton, Proctor & Kutz 2016), via sequential thresholded least squares.
# ---------------------------------------------------------------------------


def _polynomial_library(X: np.ndarray, degree: int = 2) -> tuple[np.ndarray, list[str]]:
    """Build a small candidate-term library: bias, linear, and (if degree>=2)
    per-feature squares and pairwise products. Kept deliberately small (no
    higher-degree/trig terms) -- this is meant as an interpretable, cheap
    sparse-regression baseline, not a general SINDy library search.
    """
    n, d = X.shape
    cols = [np.ones(n)]
    names = ["1"]
    for i in range(d):
        cols.append(X[:, i])
        names.append(f"x{i}")
    if degree >= 2:
        for i in range(d):
            cols.append(X[:, i] ** 2)
            names.append(f"x{i}^2")
        for i, j in combinations(range(d), 2):
            cols.append(X[:, i] * X[:, j])
            names.append(f"x{i}*x{j}")
    return np.stack(cols, axis=1), names


@dataclass
class SINDyResult:
    coefficients: np.ndarray       # (n_terms, n_outputs)
    feature_names: list[str]
    degree: int
    threshold: float
    n_iters_used: dict = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Theta, _ = _polynomial_library(X, degree=self.degree)
        return Theta @ self.coefficients

    def active_terms(self, output_index: int) -> list[tuple[str, float]]:
        """Non-zero (term, coefficient) pairs for one output column, sorted by |coef|."""
        col = self.coefficients[:, output_index]
        terms = [(name, float(c)) for name, c in zip(self.feature_names, col) if c != 0.0]
        return sorted(terms, key=lambda t: -abs(t[1]))


def fit_sindy(X: np.ndarray, Y: np.ndarray, degree: int = 2, threshold: float = 0.1,
              max_iters: int = 10) -> SINDyResult:
    """Fit one sparse linear model per output column via sequential thresholded
    least squares (STLSQ): ordinary least squares, then repeatedly zero out
    coefficients with `|coef| < threshold` and refit on the surviving terms,
    until the active set stops changing or `max_iters` is reached.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    Theta, names = _polynomial_library(X, degree=degree)
    n_terms = Theta.shape[1]
    n_outputs = Y.shape[1]

    coefficients = np.zeros((n_terms, n_outputs))
    n_iters_used = {}
    for j in range(n_outputs):
        active = np.ones(n_terms, dtype=bool)
        xi = np.zeros(n_terms)
        it = 0
        for it in range(1, max_iters + 1):
            if not active.any():
                break
            sub = Theta[:, active]
            sol, *_ = np.linalg.lstsq(sub, Y[:, j], rcond=None)
            xi[:] = 0.0
            xi[active] = sol
            new_active = np.abs(xi) >= threshold
            if np.array_equal(new_active, active):
                active = new_active
                break
            active = new_active
        coefficients[:, j] = xi
        n_iters_used[j] = it

    return SINDyResult(coefficients=coefficients, feature_names=names, degree=degree,
                        threshold=threshold, n_iters_used=n_iters_used)


# ---------------------------------------------------------------------------
# Shared per-node regression metrics (mirrors v2ecoli's own
# `evaluate_surrogate.py::_group_metrics` convention for apples-to-apples
# comparability, at per-node instead of per-group granularity).
# ---------------------------------------------------------------------------


def per_node_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                      names: list[str] | None = None) -> dict:
    """Per-column R2/RMSE, plus a `_group_metrics`-style summary."""
    Y_true = Y_true if Y_true.ndim > 1 else Y_true[:, None]
    Y_pred = Y_pred if Y_pred.ndim > 1 else Y_pred[:, None]
    d = Y_true.shape[1]
    names = names or [f"x{i}" for i in range(d)]

    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=0)
    mu = Y_true.mean(axis=0)
    ss_tot = np.sum((Y_true - mu) ** 2, axis=0)
    valid = ss_tot > 0

    r2 = np.full(d, np.nan)
    r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2, axis=0))

    per_node = {
        names[i]: {"r2": float(r2[i]), "rmse": float(rmse[i])}
        for i in range(d)
    }
    valid_r2 = r2[valid]
    summary = {
        "median_r2": float(np.median(valid_r2)) if valid_r2.size else float("nan"),
        "frac_r2_above_0.5": float(np.mean(valid_r2 > 0.5)) if valid_r2.size else float("nan"),
        "median_rmse": float(np.median(rmse)),
    }
    return {"per_node": per_node, "summary": summary}
