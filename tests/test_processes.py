"""Tests for PennyLaneAdversarialProcess."""

import numpy as np
import pytest
import torch
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results


@pytest.fixture
def core():
    core = allocate_core()
    core.register_link("ram-emitter", RAMEmitter)
    return core


def test_instantiation(core):
    """Process can be instantiated with default config."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    proc = PennyLaneAdversarialProcess(config={}, core=core)
    assert proc is not None


def test_inputs_outputs_schema(core):
    """Inputs and outputs return dicts with expected keys."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    proc = PennyLaneAdversarialProcess(config={}, core=core)
    inputs = proc.inputs()
    outputs = proc.outputs()

    assert isinstance(inputs, dict)
    assert "train_images" in inputs
    assert "train_labels" in inputs
    assert "test_images" in inputs
    assert "test_labels" in inputs
    assert isinstance(outputs, dict)
    assert "phase" in outputs
    assert "accuracy" in outputs
    assert "loss" in outputs
    assert "benign_accuracy" in outputs
    assert "adversarial_accuracy" in outputs


def test_initial_state(core):
    """Initial state has correct structure."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    proc = PennyLaneAdversarialProcess(config={}, core=core)
    state = proc.initial_state()
    assert state["phase"] == "init"
    assert state["epoch"] == 0
    assert state["accuracy"] == 0.0


def _make_synthetic_data(n_train=20, n_test=10, input_dim=16, n_classes=4, seed=42):
    """Generate synthetic classification data for testing."""
    torch.manual_seed(seed)
    X_train = torch.rand(n_train, input_dim).numpy()
    Y_train = torch.randint(0, n_classes, (n_train,)).numpy()
    X_test = torch.rand(n_test, input_dim).numpy()
    Y_test = torch.randint(0, n_classes, (n_test,)).numpy()
    return X_train, Y_train, X_test, Y_test


def _plusminus_subset(n_train=20, n_test=10):
    """Load a small subset of the PlusMinus dataset."""
    import pennylane as qml
    from pennylane import numpy as pnp
    [pm] = qml.data.load("other", name="plus-minus")
    X_train = np.array(pm.img_train[:n_train].reshape(n_train, -1))
    Y_train = pm.labels_train[:n_train]
    X_test = np.array(pm.img_test[:n_test].reshape(n_test, -1))
    Y_test = pm.labels_test[:n_test]
    return X_train, Y_train, X_test, Y_test


@pytest.mark.timeout(300)
def test_full_pipeline_synthetic(core):
    """Run full pipeline with synthetic data passed through input ports."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    X_train, Y_train, X_test, Y_test = _make_synthetic_data(
        n_train=12, n_test=6, input_dim=16, n_classes=3
    )

    config = {
        "num_qubits": 4,
        "num_layers": 4,
        "training_epochs": 1,
        "adversarial_epochs": 1,
        "pgd_iter": 2,
        "batch_size": 6,
        "seed": 42,
    }

    proc = PennyLaneAdversarialProcess(config=config, core=core)
    state = proc.initial_state()
    state["train_images"] = X_train
    state["train_labels"] = Y_train
    state["test_images"] = X_test
    state["test_labels"] = Y_test

    MAX_STEPS = 20
    for _step in range(MAX_STEPS):
        result = proc.update(state, interval=1.0)
        state.update(result)
        if state.get("phase") == "done":
            break

    assert state["phase"] == "done", (
        f"Pipeline did not reach 'done' phase; got '{state.get('phase')}' "
        f"at step {_step}"
    )
    assert state["n_queries"] > 0
    assert 0.0 <= state["benign_accuracy"] <= 1.0
    assert 0.0 <= state["adversarial_accuracy"] <= 1.0
    assert 0.0 <= state["robust_accuracy"] <= 1.0


@pytest.mark.timeout(300)
def test_full_pipeline_plusminus(core):
    """Run full pipeline with PlusMinus dataset (backward-compat fallback)."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    config = {
        "num_qubits": 4,
        "num_layers": 4,
        "training_epochs": 1,
        "adversarial_epochs": 1,
        "pgd_iter": 2,
        "batch_size": 10,
        "seed": 42,
    }

    proc = PennyLaneAdversarialProcess(config=config, core=core)
    state = proc.initial_state()

    MAX_STEPS = 20
    for _step in range(MAX_STEPS):
        result = proc.update(state, interval=1.0)
        state.update(result)
        if state.get("phase") == "done":
            break

    assert state["phase"] == "done"
    assert state["n_queries"] > 0


@pytest.mark.timeout(300)
def test_composite_assembly_with_data(core):
    """Composite with wired data stores assembles and runs."""
    from pbg_pennylane_adversarial.composites.adversarial import _build_adversarial_document

    X_train, Y_train, X_test, Y_test = _make_synthetic_data(
        n_train=12, n_test=6, input_dim=16, n_classes=3
    )

    doc = _build_adversarial_document(
        config={
            "num_qubits": 4,
            "num_layers": 4,
            "training_epochs": 1,
            "adversarial_epochs": 1,
            "pgd_iter": 2,
            "batch_size": 6,
            "seed": 42,
        },
        train_images=X_train.tolist(),
        train_labels=Y_train.tolist(),
        test_images=X_test.tolist(),
        test_labels=Y_test.tolist(),
    )

    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)

    results = gather_emitter_results(sim)
    assert results, "No emitter results collected"
    all_records = results.get(("emitter",), [])
    assert len(all_records) > 0


@pytest.mark.timeout(300)
def test_composite_assembly_plusminus(core):
    """Composite with default PlusMinus data fallback assembles and runs."""
    from pbg_pennylane_adversarial.composites.adversarial import _build_adversarial_document

    doc = _build_adversarial_document({
        "num_qubits": 4,
        "num_layers": 4,
        "training_epochs": 1,
        "adversarial_epochs": 1,
        "pgd_iter": 2,
        "n_train": 20,
        "n_test": 10,
        "batch_size": 10,
        "seed": 42,
    })

    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)

    results = gather_emitter_results(sim)
    assert results, "No emitter results collected"
    all_records = results.get(("emitter",), [])
    assert len(all_records) > 0


def test_generator_is_registered():
    """Composite generator is discoverable via the registry."""
    from pbg_pennylane_adversarial.composites import adversarial  # noqa: F401
    from pbg_superpowers.composite_generator import _REGISTRY

    matches = [
        eid for eid in _REGISTRY
        if "pennylane_adversarial" in eid
    ]
    assert matches, (
        f"No pennylane_adversarial generators found in registry; "
        f"have: {list(_REGISTRY.keys())[:10]}"
    )
