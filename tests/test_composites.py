"""Tests for PennyLane adversarial composite generators."""

import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter


def _core():
    core = allocate_core()
    core.register_link("ram-emitter", RAMEmitter)
    return core


@pytest.mark.timeout(300)
def test_adversarial_baseline_generator():
    """adversarial_baseline generator produces a runnable composite."""
    from pbg_pennylane_adversarial.composites.adversarial import adversarial_baseline

    doc = adversarial_baseline(
        core=None,
        training_epochs=1,
        epsilon=0.1,
        num_qubits=4,
    )
    core = _core()
    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)

    from process_bigraph.emitter import gather_emitter_results
    results = gather_emitter_results(sim)
    assert results, "No emitter results"


@pytest.mark.timeout(300)
def test_adversarial_lightweight_generator():
    """adversarial_lightweight generator produces a runnable composite."""
    from pbg_pennylane_adversarial.composites.adversarial import adversarial_lightweight

    doc = adversarial_lightweight(
        core=None,
        num_qubits=4,
        num_layers=4,
        n_train=20,
    )
    core = _core()
    # Override config for fast test
    doc["adversarial"]["config"]["training_epochs"] = 1
    doc["adversarial"]["config"]["adversarial_epochs"] = 1
    doc["adversarial"]["config"]["pgd_iter"] = 2
    doc["adversarial"]["config"]["n_test"] = 10
    doc["adversarial"]["config"]["batch_size"] = 10
    doc["adversarial"]["config"]["seed"] = 42

    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)

    from process_bigraph.emitter import gather_emitter_results
    results = gather_emitter_results(sim)
    assert results, "No emitter results"
