# pbg-pennylane-adversarial

Process-bigraph wrapper for adversarial attacks on quantum machine learning classifiers built with PennyLane.

Implements the PGD (Projected Gradient Descent) attack and adversarial retraining pipeline from the [PennyLane adversarial attacks tutorial](https://pennylane.ai/demos/tutorial_adversarial_attacks_QML).

## Installation

### From PyPI (recommended)

```bash
pip install pbg-pennylane-adversarial
```

Or with uv:

```bash
uv pip install pbg-pennylane-adversarial
```

### For development

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Once installed, processes register automatically via `bigraph_schema.package.discover` — no manual `register_link()` calls are needed.

## Quick Start

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results
from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

core = allocate_core()
core.register_link("ram-emitter", RAMEmitter)

proc = PennyLaneAdversarialProcess(config={
    "num_qubits": 4,
    "num_layers": 4,
    "training_epochs": 2,
    "adversarial_epochs": 1,
    "pgd_iter": 5,
    "n_train": 50,
    "n_test": 20,
}, core=core)

state = proc.initial_state()
for _ in range(20):
    result = proc.update(state, interval=1.0)
    state.update(result)
    if state["phase"] == "done":
        break

print(f"Benign accuracy: {state['benign_accuracy']:.1%}")
print(f"Adversarial accuracy: {state['adversarial_accuracy']:.1%}")
print(f"Robust accuracy: {state['robust_accuracy']:.1%}")
```

### Via composite generator

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_pennylane_adversarial.composites.adversarial import adversarial_baseline

core = allocate_core()
core.register_link("ram-emitter", RAMEmitter)

doc = adversarial_baseline(core=None, training_epochs=2, epsilon=0.1)
sim = Composite({"state": doc}, core=core)
sim.run(10.0)

from process_bigraph.emitter import gather_emitter_results
results = gather_emitter_results(sim)
```

## Pipeline Phases

| Phase | Description |
|---|---|
| `init` | Initialization |
| `training` | Supervised training on PlusMinus dataset |
| `benign_eval` | Accuracy evaluation on clean test data |
| `attack` | PGD adversarial attack generation |
| `adversarial_training` | Retraining with augmented adversarial dataset |
| `done` | Pipeline complete |

## API Reference

### `PennyLaneAdversarialProcess(config, core)`

| Config Key | Type | Default | Description |
|---|---|---|---|
| `num_qubits` | int | 8 | Number of qubits |
| `num_layers` | int | 32 | StronglyEntanglingLayers count |
| `num_reup` | int | 3 | Data-reupload multiplier |
| `output_dim` | int | 4 | Number of classes |
| `seed` | int | 1337 | Random seed |
| `learning_rate` | float | 0.1 | Adam learning rate |
| `training_epochs` | int | 4 | Training epochs |
| `batch_size` | int | 20 | Batch size |
| `epsilon` | float | 0.1 | PGD L_inf bound |
| `pgd_alpha` | float | 0.01 | PGD step size |
| `pgd_iter` | int | 10 | PGD iterations |
| `adversarial_epochs` | int | 2 | Adv. retraining epochs |
| `n_train` | int | 200 | Training samples |
| `n_test` | int | 50 | Test samples |

**Output ports:** `phase`, `epoch`, `loss`, `accuracy`, `benign_accuracy`, `adversarial_accuracy`, `robust_accuracy`, `adversarial_accuracy_drop`, `n_queries`

### Composite Generators

| Generator | Description |
|---|---|
| `adversarial_baseline` | Full pipeline, default params |
| `adversarial_robust` | Stronger attack, more retraining |
| `adversarial_lightweight` | Small circuit, fast iteration |

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
```

Opens a self-contained HTML report with Plotly charts, metrics, and architecture diagram.

## Architecture

The wrapper bridges PennyLane's `StronglyEntanglingLayers` circuit template and `TorchLayer` PyTorch integration. The Process internally manages a PyTorch `nn.Module` wrapping a PennyLane QNode. Each `update()` advances the pipeline by one epoch or evaluation phase, emitting accuracy/loss metrics through PBG output ports.

## Limitations

- Uses the PlusMinus dataset from PennyLane Datasets (internet connection required for first load; data is cached afterward).
- Requires `lightning.qubit` device (PennyLane-Lightning plugin) for fast simulation.
- Circuit evaluation is the throughput bottleneck — 256 input features × 8 qubits × 32 layers.
