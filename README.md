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

## Applications

### Cell-cycle phase classification from whole-cell model trajectories

The dataset `baseline_classification_v1.csv` is a sampled subset of **real raw
timeseries output** from a vEcoli single-cell simulation (`sms_single`,
2529 timesteps, 42 min cell cycle, 7 Parquet batch files under
`vecoli_data/outputs/sms_single/`). Each row is a 1-second timestep from the
simulated cell's trajectory.

The target column `cell_cycle_phase` classifies each timestep into one of three
stages based on the chromosome replication state directly observed in the WCM:

| Phase | Heuristic | WCM signature |
|---|---|---|
| `initiation_phase` (0) | Pre-chromosome completion | `number_of_oric = 2`, `full_chromosome = 1` |
| `chromosome_complete` (1) | Replication finished, pre-oriC duplication | `number_of_oric = 2`, `full_chromosome = 2` |
| `post_oric_duplication` (2) | Origins duplicated, late cell cycle | `number_of_oric = 4` |

This is biologically novel — most cell-cycle classifiers use just mass or oriC
count; here the **two-step replication signature** (chromosome completion at
*t* ≈ 1329 s, oriC duplication at *t* ≈ 1625 s) provides a richer target that
tests whether a quantum classifier can resolve distinct sub-phases of
chromosome replication from molecular profiles alone.

**The adversarial question:** *can a quantum classifier robustly distinguish
these three cell-cycle stages from molecular profiles alone, even under feature
perturbation?*

### Stress-response classification from whole-cell model trajectories

The dataset `baseline_classification.csv` classifies simulation timepoints into
three resource-allocation regimes using scalar features from a whole-cell
E. coli model (v2ecoli):

| Regime | Heuristic |
|---|---|
| `balanced_growth` (0) | ppGpp < 10 µM, charged tRNA > 0.75 |
| `stringent_response` (1) | ppGpp ≥ 10 µM or charged tRNA < 0.75 |
| `division_competent` (2) | `dry_mass > 600` fg, `number_of_oric ≥ 4`, healthy growth |

These two example datasets illustrate the range of biologically meaningful
classification tasks the pipeline supports — from cell-cycle staging to
physiological stress detection. Both artifacts were produced by the
`adversarial datasets format` CLI command and are fully compatible with
`PennyLaneAdversarialProcess` via `load_formatted` → `_build_adversarial_document`.


## Honest Assesment

The quantum classifier itself is unlikely to outperform classical models on WCM data. With 4 qubits, 8 layers, and ~40 trainable parameters, it has less capacity than a simple 2-layer neural net. The ~47% accuracy on 3-class v1 data (vs 33% random) confirms it learns something, but a random forest or logistic regression on the same 17 features would likely score higher with less tuning.
However, the adversarial robustness framework is genuinely novel and useful for biological modeling in three ways:
1. Measurement noise as a first-class concern (epsilon=0.05): Wet-lab measurements of ppGpp, growth rate, mass fractions, and RNA counts all carry ±5% experimental error. The PGD attack finds the worst-case misclassification within that noise bound. A model that maintains accuracy under PGD is one whose predictions are stable under realistic measurement uncertainty — a property no standard classifier (quantum or classical) guarantees.
2. Noise-tolerance, not evasion: In security, adversarial robustness prevents an attacker from evading a classifier. In biology, it means "if I re-measure this cell, will I get the same phenotype call?" This is a quality metric for the biomarker signature itself — features that survive epsilon=0.05 perturbation without changing the predicted class are diagnostically robust.
3. The "adversarial accuracy drop" as a biological signal: A large drop between benign and adversarial accuracy means the decision boundary has a fragile orientation relative to the noise directions. In biological terms, some phenotypes might be separated by thin, noisy feature margins (e.g., a single RNA species) while others are separated by broad, stable margins (e.g., mass ratios). The framework quantifies this per-dataset.
What would make this compelling
The biggest gap is No classical baseline in the pipeline. To make a convincing case, the pipeline (or at minimum the report) should run sklearn.linear_model.LogisticRegression and sklearn.ensemble.RandomForestClassifier on the same train/test split and report their benign/adversarial accuracy alongside the quantum model's. If the quantum model matches or approaches classical accuracy, the robustness framing is a bonus. If it's far behind, the novelty claim rests entirely on "quantum classifier with noise-robustness guarantee" which is a very niche sell for WCM biologists.

## Theoretical References

Academic and publication references consulted in building this package and in scoping its follow-on work (`USE_CASES.md`, `todo.md`). Grouped by what they ground.

### Core pipeline (adversarial attacks on QML)

- Wendlinger, M., Tscharke, K., & Debus, C. (2024). *A Comparative Analysis of Adversarial Robustness for Quantum and Classical Machine Learning Models.* [arXiv:2404.16154](https://arxiv.org/abs/2404.16154) — primary theoretical basis for the [PennyLane adversarial attacks tutorial](https://pennylane.ai/demos/tutorial_adversarial_attacks_QML) this repo wraps.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples.* [arXiv:1412.6572](https://arxiv.org/abs/1412.6572) — foundational adversarial-examples paper cited by the tutorial.
- Liu, Y., Arunachalam, S., & Temme, K. (2020). *A Rigorous and Robust Quantum Speed-up in Supervised Machine Learning.* [arXiv:2010.02174](https://arxiv.org/abs/2010.02174)
- Lu, S., Duan, L.-M., & Deng, D.-L. (2019). *Quantum Adversarial Machine Learning.* [arXiv:2001.00030](https://arxiv.org/abs/2001.00030)

### B1 (quantum-kernel pre-screening) planning references

- Huang, H.-Y., Broughton, M., Mohseni, M., Babbush, R., Boixo, S., Neven, H., & McClean, J. R. (2021). *Power of data in quantum machine learning.* Nature Communications 12, 2631. [arXiv:2011.01938](https://arxiv.org/abs/2011.01938) — source of the geometric-difference metric, per PennyLane's [pre-screening demo](https://pennylane.ai/qml/demos/tutorial_huang_geometric_kernel_difference).

### A1 (QGRNN-style graph-structured surrogate) planning references

- Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. (2019). *Quantum Graph Neural Networks.* [arXiv:1909.12264](https://arxiv.org/abs/1909.12264) — introduces QGNN/QGRNN, basis of PennyLane's [`tutorial_qgrnn`](https://pennylane.ai/qml/demos/tutorial_qgrnn).
- Scott, M., Gunderson, C. W., Mateescu, E. M., Zhang, Z., & Hwa, T. (2010). *Interdependence of Cell Growth and Gene Expression: Origins and Consequences.* Science 330(6007), 1099–1102. — bacterial growth-law relationship (ribosomal/RNA mass fraction scales with growth rate) used as A1's "real test" validation target.
- Cooper, S., & Helmstetter, C. E. (1968). *Chromosome replication and the division cycle of Escherichia coli B/r.* J. Mol. Biol. 31(3), 519–540. — replication-timing model underlying the `dna_mass`↔`number_of_oric` coupling check.
- Schmid, P. J. (2010). *Dynamic mode decomposition of numerical and experimental data.* Journal of Fluid Mechanics 656, 5–28. — basis for A1's linear DMD baseline arm.

### B4 (barren-plateau diagnostics) planning references

- McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications 9, 4812. — basis for PennyLane's [`tutorial_barren_plateaus`](https://pennylane.ai/qml/demos/tutorial_barren_plateaus).
- Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021). *Cost function dependent barren plateaus in shallow parametrized quantum circuits.* Nature Communications 12, 1791. — basis for the stretch-goal local-cost-function mitigation (`tutorial_local_cost_functions`).

### A1 novelty/prior-art literature check (conducted 2026-07-02, phase 0 of implementation)

Searched to determine whether QGRNN-style circuits have already been applied to real biological time-series/dynamics data, before committing engineering time to A1's novelty claim. Closest prior art found:

- **Sohail, M. A., Sudharshan, R. R., Pradhan, S. S., & Rao, A. (2026).** *Quantum Hamiltonian Learning using Time-Resolved Measurement Data and its Application to Gene Regulatory Network Inference.* [arXiv:2602.19496](https://arxiv.org/abs/2602.19496), also on [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.05.709897v1). **This is a materially close prior-art hit**: a parameterized-Hamiltonian model (QHGM) encoding gene interactions, trained via a variational learning algorithm with finite-sample recovery guarantees, applied to real Glioblastoma single-cell RNA-seq pseudotime data — recovering biologically plausible regulatory connections. It predates this session by several months. It targets gene regulatory networks from scRNA-seq pseudotime, not a mechanistic whole-cell-model's multi-observable transition dynamics, and (from the abstract alone) it isn't confirmed whether QHGM is executed as an actual Trotterized circuit (QGRNN-style) versus a classical estimator built on quantum-Hamiltonian-learning theory — the full text needs a closer read before finalizing A1's novelty claim in any write-up. At minimum, this must be cited as related work; the "first application to real biological time-series data" framing in `todo.md` needs revising to a narrower, still-checkable claim (e.g., first application to a *mechanistic whole-cell simulator's* multi-observable dynamics, as opposed to transcriptomic pseudotime).
- Additional related work surfaced, less directly overlapping: Zhang et al., *Quantum gene regulatory networks*, [arXiv:2206.15362](https://arxiv.org/abs/2206.15362); *QGHNN: A quantum graph Hamiltonian neural network*, [arXiv:2501.07986](https://arxiv.org/abs/2501.07986); *Bayesian Networks based Hybrid Quantum-Classical Machine Learning Approach to Elucidate Gene Regulatory Pathways*, [arXiv:1901.10557](https://arxiv.org/abs/1901.10557); *Quantum Deep Learning Pipeline for Next Generation Network Biology*, [bioRxiv 2025.10.28.685074](https://www.biorxiv.org/content/10.1101/2025.10.28.685074); *Feature Prediction in Quantum Graph Recurrent Neural Networks with Applications in Information Hiding*, [arXiv:2506.23144](https://arxiv.org/abs/2506.23144) (non-biological application of QGRNN); *From Graphs to Qubits: A Critical Review of Quantum Graph Neural Networks*, [arXiv:2408.06524](https://arxiv.org/abs/2408.06524) (survey).
- None of the above apply a QGRNN-style circuit to a mechanistic whole-cell-model simulator's transition dynamics specifically — the narrower claim in `todo.md`'s "Novelty — deep dive" section still appears to hold, but should be stated relative to the QHGM paper above, not as if no related work exists.