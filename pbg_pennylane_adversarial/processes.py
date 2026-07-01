"""PennyLane adversarial attack process-bigraph wrappers.

Wraps the QML classifier + PGD attack + adversarial retraining pipeline
from the PennyLane adversarial attacks tutorial.
"""

import math
import torch
from process_bigraph import Process


class PennyLaneAdversarialProcess(Process):
    """PennyLane QML classifier with PGD adversarial attack and robust training.

    Bridges the real PennyLane adversarial attacks pipeline:

        1. Build a data-reuploading QML classifier (StronglyEntanglingLayers
           + TorchLayer).
        2. Train on the PlusMinus dataset.
        3. Evaluate benign accuracy.
        4. Run a PGD (projected gradient descent) adversarial attack.
        5. Evaluate adversarial accuracy.
        6. Adversarial retraining (data augmentation with perturbed samples).
        7. Evaluate robust accuracy.

    Each ``update()`` advances the pipeline by one epoch (or one evaluation
    phase). The current ``phase`` is reported through the ``phase`` output
    port so a sibling process or emitter can track progress.

    Config
    ------
    num_qubits : int
        Number of qubits in circuit (default 8).
    num_layers : int
        Number of StronglyEntanglingLayers (default 32).
    num_reup : int
        Number of times input is repeated in data-reuploading (default 3).
    output_dim : int
        Number of output classes (default 4).
    seed : int
        Random seed for reproducibility (default 1337).
    learning_rate : float
        Adam learning rate (default 0.1).
    training_epochs : int
        Number of training epochs (default 4).
    batch_size : int
        Batch size for training (default 20).
    epsilon : float
        PGD perturbation bound L_inf (default 0.1).
    pgd_alpha : float
        PGD step size (default 0.01).
    pgd_iter : int
        PGD iterations (default 10).
    adversarial_epochs : int
        Adversarial retraining epochs (default 2).
    n_train : int
        Number of training samples to use (default 200).
    n_test : int
        Number of test samples to use (default 50).

    Inputs
    ------
    (none — data is loaded internally from PennyLane datasets)

    Outputs
    -------
    phase : string
        Current pipeline phase label.
    epoch : integer
        Current epoch within the active phase.
    loss : float
        Training or validation loss.
    accuracy : float
        Classification accuracy on current dataset.
    benign_accuracy : float
        Accuracy on benign (clean) test data.
    adversarial_accuracy : float
        Accuracy under PGD attack.
    robust_accuracy : float
        Accuracy after adversarial retraining under attack.
    adversarial_accuracy_drop : float
        Drop in accuracy from benign to adversarial (negative = worse under attack).
    n_queries : integer
        Cumulative number of circuit evaluations.
    """

    config_schema = {
        "num_qubits": {"_type": "integer", "_default": 8, "_minimum": 1},
        "num_layers": {"_type": "integer", "_default": 32, "_minimum": 1},
        "output_dim": {"_type": "integer", "_default": 4, "_minimum": 2},
        "seed": {"_type": "integer", "_default": 1337},
        "learning_rate": {"_type": "float", "_default": 0.1, "_minimum": 0.0},
        "training_epochs": {"_type": "integer", "_default": 4, "_minimum": 0},
        "batch_size": {"_type": "integer", "_default": 20, "_minimum": 1},
        "epsilon": {"_type": "float", "_default": 0.1, "_minimum": 0.0},
        "pgd_alpha": {"_type": "float", "_default": 0.01, "_minimum": 0.0},
        "pgd_iter": {"_type": "integer", "_default": 10, "_minimum": 1},
        "adversarial_epochs": {"_type": "integer", "_default": 2, "_minimum": 0},
        "n_train": {"_type": "integer", "_default": 200, "_minimum": 1},
        "n_test": {"_type": "integer", "_default": 50, "_minimum": 1},
    }

    CREATES_INTERNAL_STATE = True  # signal to PBG that state is managed internally

    def inputs(self):
        return {}

    def outputs(self):
        return {
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

    def initial_state(self):
        return {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "benign_accuracy": 0.0,
            "adversarial_accuracy": 0.0,
            "robust_accuracy": 0.0,
            "adversarial_accuracy_drop": 0.0,
            "n_queries": 0,
        }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._model = None
        self._loss_fn = None
        self._optimizer = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._data_loaded = False
        self._batch_indices = None
        self._batch_index = 0
        self._n_queries = 0
        self._benign_accuracy = 0.0
        self._adversarial_accuracy = 0.0
        self._perturbed_test = None
        self._X_adv_train = None
        self._Y_adv_train = None
        self._adv_dataset_built = False

    def _load_data(self):
        import pennylane as qml
        from pennylane import numpy as np
        import torch

        device = torch.device("cpu")
        torch.manual_seed(self.config["seed"])

        [pm] = qml.data.load("other", name="plus-minus")
        n_train = self.config["n_train"]
        n_test = self.config["n_test"]

        X_train_np = pm.img_train[:n_train].reshape(n_train, -1)
        X_test_np = pm.img_test[:n_test].reshape(n_test, -1)
        Y_train_np = pm.labels_train[:n_train]
        Y_test_np = pm.labels_test[:n_test]

        self._X_train = torch.from_numpy(np.array(X_train_np)).float().to(device)
        self._X_test = torch.from_numpy(np.array(X_test_np)).float().to(device)
        self._Y_train = torch.from_numpy(np.array(Y_train_np)).long().to(device)
        self._Y_test = torch.from_numpy(np.array(Y_test_np)).long().to(device)
        self._data_loaded = True

    def _build_model(self):
        import torch
        import pennylane as qml

        torch.manual_seed(self.config["seed"])
        c = self.config
        device = torch.device("cpu")

        num_qubits = c["num_qubits"]
        num_layers = c["num_layers"]
        output_dim = c["output_dim"]

        weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=num_layers, n_wires=num_qubits
        )
        weights_elements = weights_shape[0] * weights_shape[1] * weights_shape[2]

        # Auto-compute num_reup to satisfy the dimensional constraint:
        # num_reup * input_dim == weights_elements (so the data-reuploading
        # tensor can be reshaped to weights_shape).
        input_dim = self._X_train.shape[1]
        num_reup = max(1, weights_elements // input_dim)
        if num_reup * input_dim < weights_elements:
            num_reup += 1

        dev = qml.device("lightning.qubit", wires=num_qubits)

        @qml.qnode(dev)
        def circuit(inputs, weights, bias):
            inputs = torch.reshape(inputs, weights_shape)
            qml.StronglyEntanglingLayers(
                weights=weights * inputs + bias, wires=range(num_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(output_dim)]

        param_shapes = {"weights": weights_shape, "bias": weights_shape}
        init_vals = {
            "weights": 0.1 * torch.rand(weights_shape),
            "bias": 0.1 * torch.rand(weights_shape),
        }

        class _QMLModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_reup = num_reup
                self.weights_elements = weights_elements
                self.qcircuit = qml.qnn.TorchLayer(
                    qnode=circuit, weight_shapes=param_shapes, init_method=init_vals
                )

            def forward(self, x):
                repeated = torch.hstack([x] * self.num_reup)
                trimmed = repeated[:self.weights_elements]
                return self.qcircuit(trimmed)

        self._model = _QMLModule().to(device)
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=c["learning_rate"]
        )

    def _accuracy(self, labels, predictions):
        acc = 0.0
        for l, p in zip(labels, predictions):
            if torch.argmax(p) == l:
                acc += 1.0
        return acc / len(labels)

    def _gen_batches(self, num_samples, num_batches):
        import torch
        assert num_samples % num_batches == 0
        perm_ind = torch.reshape(torch.randperm(num_samples), (num_batches, -1))
        return perm_ind

    def _pgd_attack(self, feats, labels, epsilon, alpha, num_iter):
        import torch
        delta = torch.zeros_like(feats, requires_grad=True)
        for t in range(num_iter):
            feats_adv = feats + delta
            outputs = [self._model(f) for f in feats_adv]
            l = self._loss_fn(torch.stack(outputs), labels)
            l.backward()
            delta.data = (
                delta + alpha * delta.grad.detach().sign()
            ).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        self._n_queries += num_iter * len(feats)
        return delta.detach()

    def update(self, state, interval):
        c = self.config

        if not self._data_loaded:
            self._load_data()
        if self._model is None:
            self._build_model()

        phase = state.get("phase", "init")
        epoch = state.get("epoch", 0)
        loss_val = 0.0
        acc_val = 0.0
        n_queries = self._n_queries

        if phase == "init":
            # Transition to training
            self._n_queries = 0
            return {
                "phase": "training",
                "epoch": 0,
                "loss": 0.0,
                "accuracy": 0.0,
                "benign_accuracy": 0.0,
                "adversarial_accuracy": 0.0,
                "robust_accuracy": 0.0,
                "adversarial_accuracy_drop": 0.0,
                "n_queries": 0,
            }

        elif phase == "training":
            if epoch < c["training_epochs"]:
                self._model.train()
                n_train = self._X_train.shape[0]
                num_batches = n_train // c["batch_size"]
                batch_ind = self._gen_batches(n_train, num_batches)
                total_loss = 0.0

                for it in range(num_batches):
                    self._optimizer.zero_grad()
                    feats_batch = self._X_train[batch_ind[it]]
                    labels_batch = self._Y_train[batch_ind[it]]
                    outputs = [self._model(f) for f in feats_batch]
                    batch_loss = self._loss_fn(torch.stack(outputs), labels_batch)
                    batch_loss.backward()
                    self._optimizer.step()
                    total_loss += batch_loss.item()
                    self._n_queries += len(feats_batch)

                # Evaluate on subset
                self._model.eval()
                with torch.no_grad():
                    preds_train = [self._model(f) for f in self._X_train[:50]]
                    acc_train = self._accuracy(self._Y_train[:50], preds_train)

                return {
                    "phase": "training",
                    "epoch": epoch + 1,
                    "loss": total_loss / num_batches,
                    "accuracy": acc_train,
                    "benign_accuracy": self._benign_accuracy,
                    "adversarial_accuracy": self._adversarial_accuracy,
                    "robust_accuracy": 0.0,
                    "adversarial_accuracy_drop": 0.0,
                    "n_queries": self._n_queries,
                }
            else:
                # Training complete, evaluate benign
                self._model.eval()
                with torch.no_grad():
                    preds_test = [self._model(f) for f in self._X_test]
                    benign_acc = self._accuracy(self._Y_test, preds_test)
                self._benign_accuracy = benign_acc
                return {
                    "phase": "benign_eval",
                    "epoch": 0,
                    "loss": 0.0,
                    "accuracy": benign_acc,
                    "benign_accuracy": benign_acc,
                    "adversarial_accuracy": 0.0,
                    "robust_accuracy": 0.0,
                    "adversarial_accuracy_drop": 0.0,
                    "n_queries": self._n_queries,
                }

        elif phase == "benign_eval":
            # Run PGD attack
            self._model.eval()
            perturbations = self._pgd_attack(
                self._X_test,
                self._Y_test,
                epsilon=c["epsilon"],
                alpha=c["pgd_alpha"],
                num_iter=c["pgd_iter"],
            )
            self._perturbed_test = self._X_test + perturbations

            self._model.eval()
            with torch.no_grad():
                adv_preds = [self._model(f) for f in self._perturbed_test]
                adv_acc = self._accuracy(self._Y_test, adv_preds)
            self._adversarial_accuracy = adv_acc
            drop = self._benign_accuracy - adv_acc

            return {
                "phase": "attack",
                "epoch": 0,
                "loss": 0.0,
                "accuracy": adv_acc,
                "benign_accuracy": self._benign_accuracy,
                "adversarial_accuracy": adv_acc,
                "robust_accuracy": 0.0,
                "adversarial_accuracy_drop": drop,
                "n_queries": self._n_queries,
            }

        elif phase == "attack":
            # Build adversarial training dataset
            self._model.train()
            if not self._adv_dataset_built:
                adv_delta = self._pgd_attack(
                    self._X_train[:20],
                    self._Y_train[:20],
                    epsilon=c["epsilon"],
                    alpha=c["pgd_alpha"],
                    num_iter=c["pgd_iter"],
                )
                adv_train = adv_delta + self._X_train[:20]
                self._X_adv_train = torch.cat((self._X_train, adv_train))
                self._Y_adv_train = torch.cat(
                    (self._Y_train, self._Y_train[:20])
                )
                self._adv_dataset_built = True
                self._optimizer = torch.optim.Adam(
                    self._model.parameters(), lr=c["learning_rate"]
                )

            return {
                "phase": "adversarial_training",
                "epoch": 0,
                "loss": 0.0,
                "accuracy": 0.0,
                "benign_accuracy": self._benign_accuracy,
                "adversarial_accuracy": self._adversarial_accuracy,
                "robust_accuracy": 0.0,
                "adversarial_accuracy_drop": self._benign_accuracy - self._adversarial_accuracy,
                "n_queries": self._n_queries,
            }

        elif phase == "adversarial_training":
            if epoch < c["adversarial_epochs"]:
                self._model.train()
                n_adv = self._X_adv_train.shape[0]
                num_batches = n_adv // c["batch_size"]
                batch_ind = self._gen_batches(n_adv, num_batches)
                total_loss = 0.0

                for it in range(num_batches):
                    self._optimizer.zero_grad()
                    feats_batch = self._X_adv_train[batch_ind[it]]
                    labels_batch = self._Y_adv_train[batch_ind[it]]
                    outputs = [self._model(f) for f in feats_batch]
                    batch_loss = self._loss_fn(torch.stack(outputs), labels_batch)
                    batch_loss.backward()
                    self._optimizer.step()
                    total_loss += batch_loss.item()
                    self._n_queries += len(feats_batch)

                self._model.eval()
                with torch.no_grad():
                    preds_adv = [self._model(f) for f in self._X_adv_train[:50]]
                    acc_adv = self._accuracy(self._Y_adv_train[:50], preds_adv)

                return {
                    "phase": "adversarial_training",
                    "epoch": epoch + 1,
                    "loss": total_loss / num_batches,
                    "accuracy": acc_adv,
                    "benign_accuracy": self._benign_accuracy,
                    "adversarial_accuracy": self._adversarial_accuracy,
                    "robust_accuracy": 0.0,
                    "adversarial_accuracy_drop": self._benign_accuracy - self._adversarial_accuracy,
                    "n_queries": self._n_queries,
                }
            else:
                # Evaluate robust accuracy
                self._model.eval()
                with torch.no_grad():
                    robust_preds = [self._model(f) for f in self._perturbed_test]
                    robust_acc = self._accuracy(self._Y_test, robust_preds)

                return {
                    "phase": "done",
                    "epoch": 0,
                    "loss": 0.0,
                    "accuracy": robust_acc,
                    "benign_accuracy": self._benign_accuracy,
                    "adversarial_accuracy": self._adversarial_accuracy,
                    "robust_accuracy": robust_acc,
                    "adversarial_accuracy_drop": self._benign_accuracy - self._adversarial_accuracy,
                    "n_queries": self._n_queries,
                }

        elif phase == "done":
            return {
                "phase": "done",
                "epoch": 0,
                "loss": 0.0,
                "accuracy": state.get("robust_accuracy", 0.0),
                "benign_accuracy": state.get("benign_accuracy", 0.0),
                "adversarial_accuracy": state.get("adversarial_accuracy", 0.0),
                "robust_accuracy": state.get("robust_accuracy", 0.0),
                "adversarial_accuracy_drop": state.get("benign_accuracy", 0.0) - state.get("adversarial_accuracy", 0.0),
                "n_queries": state.get("n_queries", 0),
            }

        return {
            "phase": "error",
            "epoch": 0,
            "loss": 0.0,
            "accuracy": 0.0,
            "benign_accuracy": 0.0,
            "adversarial_accuracy": 0.0,
            "robust_accuracy": 0.0,
            "adversarial_accuracy_drop": 0.0,
            "n_queries": 0,
        }
