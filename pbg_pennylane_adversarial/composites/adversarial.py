"""Composite generators for the PennyLane adversarial attack pipeline.

Each generator returns a Composite document suitable for
``Composite({"state": doc}, core=core)`` that runs the full
train → attack → adversarial-retrain → evaluate pipeline.

Data is included in the composite stores so the process reads it through
its input ports. By default the PlusMinus dataset is used; callers can
pass custom ``train_images``, ``train_labels``, ``test_images``,
``test_labels`` keyword arguments to override.
"""

import numpy as np
from pbg_superpowers.composite_generator import composite_generator


def _load_plusminus(n_train=200, n_test=50):
    """Load PlusMinus dataset and return data lists suitable for stores."""
    import pennylane as qml
    from pennylane import numpy as pnp
    [pm] = qml.data.load("other", name="plus-minus")
    return (
        np.array(pm.img_train[:n_train].reshape(n_train, -1)).tolist(),
        pm.labels_train[:n_train].tolist(),
        np.array(pm.img_test[:n_test].reshape(n_test, -1)).tolist(),
        pm.labels_test[:n_test].tolist(),
    )


def _build_adversarial_document(
    config=None,
    train_images=None, train_labels=None,
    test_images=None, test_labels=None,
):
    """Build a Composite document for the PennyLane adversarial pipeline.

    Parameters
    ----------
    config : dict or None
        Overrides for PennyLaneAdversarialProcess config.
    train_images : list or None
        Training feature matrix as nested list.
    train_labels : list or None
        Training labels.
    test_images : list or None
        Test feature matrix as nested list.
    test_labels : list or None
        Test labels.

    Returns
    -------
    dict
        Composite state document with data wired through stores.
    """
    if config is None:
        config = {}

    if train_images is None:
        n_train = config.get("n_train", 200)
        n_test = config.get("n_test", 50)
        train_images, train_labels, test_images, test_labels = _load_plusminus(
            n_train, n_test
        )

    return {
        "adversarial": {
            "_type": "process",
            "address": "local:PennyLaneAdversarialProcess",
            "config": config,
            "interval": 1.0,
            "inputs": {
                "train_images": ["stores", "train_images"],
                "train_labels": ["stores", "train_labels"],
                "test_images": ["stores", "test_images"],
                "test_labels": ["stores", "test_labels"],
            },
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "accuracy": ["stores", "accuracy"],
                "benign_accuracy": ["stores", "benign_accuracy"],
                "adversarial_accuracy": ["stores", "adversarial_accuracy"],
                "robust_accuracy": ["stores", "robust_accuracy"],
                "adversarial_accuracy_drop": ["stores", "adversarial_accuracy_drop"],
                "n_queries": ["stores", "n_queries"],
            },
        },
        "stores": {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "benign_accuracy": 0.0,
            "adversarial_accuracy": 0.0,
            "robust_accuracy": 0.0,
            "adversarial_accuracy_drop": 0.0,
            "n_queries": 0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "phase": "string",
                    "epoch": "integer",
                    "loss": "float",
                    "accuracy": "float",
                    "benign_accuracy": "float",
                    "adversarial_accuracy": "float",
                    "robust_accuracy": "float",
                    "adversarial_accuracy_drop": "float",
                    "n_queries": "integer",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "accuracy": ["stores", "accuracy"],
                "benign_accuracy": ["stores", "benign_accuracy"],
                "adversarial_accuracy": ["stores", "adversarial_accuracy"],
                "robust_accuracy": ["stores", "robust_accuracy"],
                "adversarial_accuracy_drop": ["stores", "adversarial_accuracy_drop"],
                "n_queries": ["stores", "n_queries"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="pennylane_adversarial_adversarial_baseline",
    description=(
        "Full train → PGD attack → adversarial retrain pipeline with "
        "default hyperparameters (4 training epochs, epsilon=0.1, "
        "2 adversarial retraining epochs)."
    ),
    parameters={
        "training_epochs": {
            "type": "integer",
            "default": 4,
            "description": "Number of initial training epochs",
        },
        "epsilon": {
            "type": "float",
            "default": 0.1,
            "description": "PGD perturbation bound L_inf",
        },
        "num_qubits": {
            "type": "integer",
            "default": 8,
            "description": "Number of qubits in the circuit",
        },
    },
)
def adversarial_baseline(core=None, *, training_epochs=4, epsilon=0.1, num_qubits=8):
    return _build_adversarial_document({
        "training_epochs": training_epochs,
        "epsilon": epsilon,
        "num_qubits": num_qubits,
        "adversarial_epochs": 2,
    })


@composite_generator(
    name="pennylane_adversarial_adversarial_robust",
    description=(
        "Adversarial robustness pipeline with stronger attack "
        "(epsilon=0.2, 20 PGD iterations) and more adversarial "
        "retraining (4 epochs)."
    ),
    parameters={
        "epsilon": {
            "type": "float",
            "default": 0.2,
            "description": "PGD perturbation bound L_inf (stronger attack)",
        },
        "pgd_iter": {
            "type": "integer",
            "default": 20,
            "description": "Number of PGD iterations",
        },
        "adversarial_epochs": {
            "type": "integer",
            "default": 4,
            "description": "Number of adversarial retraining epochs",
        },
    },
)
def adversarial_robust(core=None, *, epsilon=0.2, pgd_iter=20, adversarial_epochs=4):
    return _build_adversarial_document({
        "epsilon": epsilon,
        "pgd_iter": pgd_iter,
        "adversarial_epochs": adversarial_epochs,
        "training_epochs": 4,
    })


@composite_generator(
    name="pennylane_adversarial_adversarial_lightweight",
    description=(
        "Lightweight pipeline with reduced circuit (4 qubits, "
        "8 layers) and fewer samples for fast iteration."
    ),
    parameters={
        "num_qubits": {
            "type": "integer",
            "default": 4,
            "description": "Number of qubits in the circuit",
        },
        "num_layers": {
            "type": "integer",
            "default": 8,
            "description": "Number of StronglyEntanglingLayers",
        },
        "n_train": {
            "type": "integer",
            "default": 50,
            "description": "Number of training samples to use",
        },
    },
)
def adversarial_lightweight(core=None, *, num_qubits=4, num_layers=8, n_train=50):
    return _build_adversarial_document({
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "n_train": n_train,
        "n_test": 20,
        "training_epochs": 2,
        "adversarial_epochs": 1,
    })
