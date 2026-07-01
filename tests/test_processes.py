"""Tests for PennyLaneAdversarialProcess."""

import pytest
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


@pytest.mark.timeout(300)
def test_full_pipeline_lightweight(core):
    """Run the full pipeline with lightweight config (few epochs, small model)."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    config = {
        "num_qubits": 4,
        "num_layers": 4,
        "training_epochs": 1,
        "adversarial_epochs": 1,
        "pgd_iter": 2,
        "n_train": 20,
        "n_test": 10,
        "batch_size": 10,
        "seed": 42,
    }

    proc = PennyLaneAdversarialProcess(config=config, core=core)
    state = proc.initial_state()

    # Run through phases
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
def test_composite_assembly_lightweight(core):
    """Composite assembles and runs without error (lightweight config)."""
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
    # Check that at least one result was emitted
    all_records = results.get(("emitter",), [])
    assert len(all_records) > 0


def test_generator_is_registered():
    """Composite generator is discoverable via the registry.

    Importing the composites subpackage triggers the @composite_generator
    decorators which populate the shared _REGISTRY.
    """
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
