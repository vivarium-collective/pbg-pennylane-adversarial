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

## Using Custom Datasets

The pipeline accepts **any classification dataset** through four input ports rather than being hardcoded to a single built-in dataset. Features are supplied as float arrays and labels as integer arrays.

### Data format requirements

| Requirement | Details |
|---|---|
| **Features** | `float64` array, shape `(n_samples, input_dim)` |
| **Labels** | `int64` array, shape `(n_samples,)` — must be 0-indexed integers `0..C-1` |
| **Task type** | Classification only (uses `CrossEntropyLoss`, classification accuracy) |
| **Min classes** | 2 (binary) |
| **Train/test split** | You provide both splits — the pipeline does not split for you |

### Two ways to supply data

#### Option 1: Direct — pass data in the state dict

```python
import numpy as np
from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

proc = PennyLaneAdversarialProcess(config={
    "num_qubits": 4,
    "num_layers": 4,
    "training_epochs": 2,
    "batch_size": 10,
}, core=core)

state = proc.initial_state()
# Wire data through input ports
state["train_images"] = np.random.randn(100, 16).tolist()   # 100 samples, 16 features
state["train_labels"] = np.random.randint(0, 3, 100).tolist()  # 3 classes
state["test_images"] = np.random.randn(20, 16).tolist()
state["test_labels"] = np.random.randint(0, 3, 20).tolist()

for _ in range(20):
    result = proc.update(state, interval=1.0)
    state.update(result)
    if state["phase"] == "done":
        break
```

The process caches data internally on the first `update()` call. On subsequent
calls it reuses the cached tensors, so there is no need to keep data in the
state dict after the first step.

#### Option 2: Composite — wire data through stores

Use the `_build_adversarial_document()` helper to create a composite document
with your data embedded in stores:

```python
import numpy as np
from pbg_pennylane_adversarial.composites.adversarial import _build_adversarial_document

doc = _build_adversarial_document(
    config={
        "num_qubits": 4,
        "num_layers": 4,
        "training_epochs": 2,
    },
    train_images=np.random.randn(100, 16).tolist(),
    train_labels=np.random.randint(0, 3, 100).tolist(),
    test_images=np.random.randn(20, 16).tolist(),
    test_labels=np.random.randint(0, 3, 20).tolist(),
)
sim = Composite({"state": doc}, core=core)
sim.run(15.0)
```

When `train_images` is `None`, `_build_adversarial_document()` falls back to
the PlusMinus dataset (backward-compatible default).

### Auto-detection of dimensions

`input_dim` and `output_dim` are **auto-detected** from the data:

- **`input_dim`** — taken from `X_train.shape[1]` (number of columns).
- **`output_dim`** — taken from `len(torch.unique(Y_train))` (number of unique
  label values).

Override them via config to force specific values:

```python
config = {
    "input_dim": 16,   # override auto-detect
    "output_dim": 3,   # override auto-detect
}
```

If `output_dim` is auto-detected and fewer than 2 unique labels are found,
it defaults to 2.

The **data-reupload multiplier** (`num_reup`) is also auto-computed to satisfy
the `StronglyEntanglingLayers` weight tensor shape constraint
(`n_layers * n_qubits * 3 == num_reup * input_dim`). You never need to set it.

### Using real-world datasets

#### From sklearn

```python
from sklearn.datasets import load_digits, train_test_split

data = load_digits()
X, y = data.data, data.target  # 1797 samples, 64 features, 10 classes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

doc = _build_adversarial_document(
    config={"num_qubits": 6, "num_layers": 8, "training_epochs": 4},
    train_images=X_train.tolist(),
    train_labels=y_train.tolist(),
    test_images=X_test.tolist(),
    test_labels=y_test.tolist(),
)
```

#### From a CSV

```python
import pandas as pd
import numpy as np

df = pd.read_csv("my_data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

proc = PennyLaneAdversarialProcess(config={
    "num_qubits": min(X.shape[1], 8),
    "num_layers": 8,
    "training_epochs": 4,
}, core=core)

state = proc.initial_state()
state["train_images"] = X_train.tolist()
state["train_labels"] = y_train.tolist()
state["test_images"] = X_test.tolist()
state["test_labels"] = y_test.tolist()
```

### PlusMinus fallback (default)

When no data is wired, the process automatically loads PennyLane's built-in
**PlusMinus** dataset (grayscale digit images, 28×28 = 784 features, 2
classes). This is backward compatible with the original behavior:

```python
# No data in state → auto-loads PlusMinus (200 train, 50 test)
state = proc.initial_state()
result = proc.update(state, interval=1.0)
```

### What does NOT change

- The **model architecture** is fixed: `StronglyEntanglingLayers` with
  `TorchLayer` and `CrossEntropyLoss`. You control `num_qubits` and
  `num_layers` via config, but the circuit structure stays the same.
- The **pipeline phases** are fixed: train → benign eval → PGD attack →
  adversarial retrain → robust eval.
- Data is always converted to list-of-lists or nested Python lists for PBG
  store serialization. NumPy arrays are accepted (`.tolist()` is called
  internally).

### Limitations

- **Classification only** — the loss function, accuracy metric, and output
  layer are designed for discrete class labels. Regression or sequence
  prediction would require new code.
- Labels must be **0-indexed contiguous integers** (`0, 1, 2, ...`).
- The PGD attack assumes continuous-valued features. Binary/categorical
  features may produce meaningless perturbations.
- Large datasets (>10k samples) will be slow due to per-sample circuit
  evaluation (the PennyLane forward pass runs one sample at a time).

## Pipeline Phases

| Phase | Description |
|---|---|
| `init` | Initialization |
| `training` | Supervised training on supplied dataset |
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
| `input_dim` | int or None | None | Feature dimension (auto-detected from data if None) |
| `output_dim` | int or None | None | Number of classes (auto-detected from labels if None) |
| `seed` | int | 1337 | Random seed |
| `learning_rate` | float | 0.1 | Adam learning rate |
| `training_epochs` | int | 4 | Training epochs |
| `batch_size` | int | 20 | Batch size |
| `epsilon` | float | 0.1 | PGD L_inf bound |
| `pgd_alpha` | float | 0.01 | PGD step size |
| `pgd_iter` | int | 10 | PGD iterations |
| `adversarial_epochs` | int | 2 | Adv. retraining epochs |

**Input ports:** `train_images` (array[float64]), `train_labels` (array[int64]),
`test_images` (array[float64]), `test_labels` (array[int64])

**Output ports:** `phase`, `epoch`, `loss`, `accuracy`, `benign_accuracy`,
`adversarial_accuracy`, `robust_accuracy`, `adversarial_accuracy_drop`, `n_queries`

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

The wrapper bridges PennyLane's `StronglyEntanglingLayers` circuit template and `TorchLayer` PyTorch integration. Data arrives through four input ports (`train_images`, `train_labels`, `test_images`, `test_labels`) and is cached internally. The Process manages a PyTorch `nn.Module` wrapping a PennyLane QNode. Each `update()` advances the pipeline by one epoch or evaluation phase, emitting accuracy/loss metrics through PBG output ports.

## Limitations

- Requires `lightning.qubit` device (PennyLane-Lightning plugin) for fast simulation.
- Circuit evaluation is the throughput bottleneck — per-sample forward pass is the dominant cost.
- The PGD attack assumes continuous-valued features.
