"""Classical (sklearn) baseline comparison for the adversarial pipeline.

Gives the report a reference point: does the quantum classifier actually add
value over `LogisticRegression`/`RandomForestClassifier` trained on the same
split? PGD (used for the QML model) needs gradients through the model; sklearn
classifiers aren't differentiable, so the adversarial perturbation here is a
one-shot FGSM-style step using a finite-difference estimate of the loss
gradient (works for any classifier exposing `predict_proba`), clipped to the
same L_inf bound `epsilon` used by the QML PGD attack.

Each baseline also gets a *transfer* accuracy: the same perturbation the QML
PGD attack produced against the quantum circuit, replayed against the
classical model's own test predictions. The white-box (own-attack) number is
the fair per-model robustness comparison; the transfer number answers whether
adversarial examples crafted against the quantum circuit generalize to
classical architectures — the two measure different things and neither
subsumes the other.
"""

from __future__ import annotations

import numpy as np


def _fgsm_perturbation(model, X: np.ndarray, y: np.ndarray, epsilon: float,
                        h: float = 1e-3) -> np.ndarray:
    rows = np.arange(len(y))
    base_loss = -np.log(np.clip(model.predict_proba(X)[rows, y], 1e-12, 1.0))

    grad = np.zeros_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        X_pert = X.copy()
        X_pert[:, j] += h
        loss_j = -np.log(np.clip(model.predict_proba(X_pert)[rows, y], 1e-12, 1.0))
        grad[:, j] = (loss_j - base_loss) / h

    return epsilon * np.sign(grad)


def _evaluate(model, X_train, y_train, X_test, y_test, epsilon: float,
              transfer_delta: np.ndarray | None) -> dict:
    model.fit(X_train, y_train)
    benign_accuracy = float(model.score(X_test, y_test))

    delta = _fgsm_perturbation(model, X_test, y_test, epsilon)
    adversarial_accuracy = float(model.score(X_test + delta, y_test))

    result = {
        "benign_accuracy": benign_accuracy,
        "adversarial_accuracy": adversarial_accuracy,
    }
    if transfer_delta is not None:
        result["transfer_adversarial_accuracy"] = float(
            model.score(X_test + transfer_delta, y_test)
        )
    return result


def run_baselines(data: dict, epsilon: float = 0.05, seed: int = 42,
                   transfer_delta: list | np.ndarray | None = None) -> dict:
    """Train/evaluate classical baselines on the same train/test split as the QML run.

    Parameters
    ----------
    data : dict
        Loaded formatted-dataset dict (``train_images``, ``train_labels``,
        ``test_images``, ``test_labels``).
    epsilon : float
        L_inf perturbation bound, matched to the QML pipeline's PGD ``epsilon``.
    seed : int
        Random state for `RandomForestClassifier`.
    transfer_delta : array-like, optional
        The QML PGD attack's perturbation of the test set (process output
        ``perturbation_delta``), shape (n_test, input_dim). When given, each
        baseline is also scored against ``test_images + transfer_delta`` to
        measure how well adversarial examples crafted against the quantum
        circuit transfer to a classical model.

    Returns
    -------
    dict
        ``{"logistic_regression": {...}, "random_forest": {...}}``, each with
        ``benign_accuracy`` / ``adversarial_accuracy`` and, if
        ``transfer_delta`` was given, ``transfer_adversarial_accuracy``.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X_train = np.asarray(data["train_images"], dtype=np.float64)
    y_train = np.asarray(data["train_labels"], dtype=np.int64)
    X_test = np.asarray(data["test_images"], dtype=np.float64)
    y_test = np.asarray(data["test_labels"], dtype=np.int64)

    transfer = (np.asarray(transfer_delta, dtype=np.float64)
                if transfer_delta is not None and len(transfer_delta) > 0 else None)

    return {
        "logistic_regression": _evaluate(
            LogisticRegression(max_iter=1000),
            X_train, y_train, X_test, y_test, epsilon, transfer,
        ),
        "random_forest": _evaluate(
            RandomForestClassifier(n_estimators=100, random_state=seed),
            X_train, y_train, X_test, y_test, epsilon, transfer,
        ),
    }
