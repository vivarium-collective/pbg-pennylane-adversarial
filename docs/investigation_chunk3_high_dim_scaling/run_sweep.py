"""Reproducible experiment sweep backing the chunk-3 (High-Dimensional Feature
Scaling) investigation report. Writes real measured results to results.json --
no numbers in the report are hand-typed; they're all loaded from that file.

Run from the repo root: uv run python docs/investigation_chunk3_high_dim_scaling/run_sweep.py

Three experiments:
  1. position_sensitivity   -- at a dim that needs no mitigation at all, does
                                WHERE the informative feature lands in the
                                reshaped weight tensor matter?
  2. redundancy_sweep       -- across input_dim, compare the genuine pre-PCA-fix
                                code (fetched live via `git show`, not a hand
                                copy) against the current code (PCA mitigation).
  3. layers_epoch_scaling   -- the rejected "auto-raise num_layers" fix: does
                                giving it more training epochs rescue it?

The pre-fix baseline is the processes.py revision immediately before the PCA
mitigation was added (commit tagged CHUNK3_PRE_FIX_REV below); update that
constant if this script is re-run long after that history has moved on.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from process_bigraph import allocate_core

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parent / "results.json"

# Commit immediately before the PCA-mitigation fix landed (chunk 3 of NEXT_STEPS.md).
CHUNK3_PRE_FIX_REV = "679fd28"

CORE = allocate_core()

# --- load the genuine pre-fix code straight from git history, not a hand-copy ---
_orig_src = subprocess.run(
    ["git", "show", f"{CHUNK3_PRE_FIX_REV}:pbg_pennylane_adversarial/processes.py"],
    cwd=REPO_ROOT, capture_output=True, text=True, check=True,
).stdout
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(_orig_src)
    _orig_path = f.name

_spec = importlib.util.spec_from_file_location("processes_pre_fix_baseline", _orig_path)
assert _spec is not None and _spec.loader is not None
_orig_mod = importlib.util.module_from_spec(_spec)
sys.modules["processes_pre_fix_baseline"] = _orig_mod
_spec.loader.exec_module(_orig_mod)
OriginalProcess = _orig_mod.PennyLaneAdversarialProcess

from pbg_pennylane_adversarial.processes import PennyLaneAdversarialProcess as CurrentProcess


def make_data(input_dim, signal_col, n=100, seed=0, amplify=4.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, input_dim)).astype(np.float64)
    X[:, signal_col] *= amplify
    y = (X[:, signal_col] > 0).astype(np.int64)
    split = int(n * 0.7)
    return {
        "train_images": X[:split].tolist(), "train_labels": y[:split].tolist(),
        "test_images": X[split:].tolist(), "test_labels": y[split:].tolist(),
    }


def run_to_benign_eval(process_cls, config, data):
    proc = process_cls(config=config, core=CORE)
    state = {**data, "phase": "init", "epoch": 0}
    state.update(proc.update(state, 1.0))
    steps = 0
    while state["phase"] != "benign_eval":
        state.update(proc.update(state, 1.0))
        steps += 1
        if steps > 250:
            raise RuntimeError("did not reach benign_eval")
    return state["benign_accuracy"]


results = {}
t0 = time.time()

# ---------------------------------------------------------------------------
# Experiment 1: position sensitivity, input_dim=16, no mitigation needed here
# (weights_elements=48 with num_qubits=4,num_layers=4; 16 is well under any
# fix threshold in either the original or current code)
# ---------------------------------------------------------------------------
print("Experiment 1: position sensitivity")
exp1 = []
SEEDS_1 = [1, 2, 3, 4, 5]
for position, col in [("first", 0), ("last", 15)]:
    for seed in SEEDS_1:
        data = make_data(input_dim=16, signal_col=col, seed=seed)
        acc = run_to_benign_eval(
            CurrentProcess,
            {"num_qubits": 4, "num_layers": 4, "training_epochs": 8,
             "batch_size": 35, "seed": seed, "learning_rate": 0.2},
            data,
        )
        exp1.append({"position": position, "seed": seed, "accuracy": acc})
        print(f"  position={position:5s} seed={seed} acc={acc:.3f}")
results["position_sensitivity"] = exp1

# ---------------------------------------------------------------------------
# Experiment 2: redundancy sweep, original (no fix) vs current (PCA fix)
# ---------------------------------------------------------------------------
print("\nExperiment 2: redundancy sweep")
exp2 = []
DIMS = [8, 16, 24, 32, 48, 64, 100]
SEEDS_2 = [1, 2, 3]
for input_dim in DIMS:
    for variant, cls in [("original_no_fix", OriginalProcess), ("current_pca_fix", CurrentProcess)]:
        for seed in SEEDS_2:
            data = make_data(input_dim=input_dim, signal_col=0, seed=seed)
            acc = run_to_benign_eval(
                cls,
                {"num_qubits": 4, "num_layers": 4, "training_epochs": 8,
                 "batch_size": 35, "seed": seed, "learning_rate": 0.2},
                data,
            )
            exp2.append({"input_dim": input_dim, "variant": variant, "seed": seed, "accuracy": acc})
            print(f"  dim={input_dim:4d} variant={variant:16s} seed={seed} acc={acc:.3f}")
results["redundancy_sweep"] = exp2
results["redundancy_sweep_meta"] = {
    "weights_elements": 48,  # num_qubits=4 * num_layers=4 * 3
    "current_min_num_reup": CurrentProcess.MIN_NUM_REUP,
    "current_pca_trigger_dim": 48 // CurrentProcess.MIN_NUM_REUP,
    "original_truncation_trigger_dim": 48,
}

# ---------------------------------------------------------------------------
# Experiment 3: rejected fix -- manually force num_layers=8 (what the
# auto-raise logic would have chosen for input_dim=48) and vary epoch budget
# ---------------------------------------------------------------------------
print("\nExperiment 3: layers+epochs scaling (rejected fix)")
exp3 = []
SEEDS_3 = [1, 2, 3]
EPOCHS = [8, 25, 50]
for epochs in EPOCHS:
    for seed in SEEDS_3:
        data = make_data(input_dim=48, signal_col=0, seed=seed)
        acc = run_to_benign_eval(
            CurrentProcess,
            {"num_qubits": 4, "num_layers": 8, "training_epochs": epochs,
             "batch_size": 35, "seed": seed, "learning_rate": 0.2},
            data,
        )
        exp3.append({"epochs": epochs, "seed": seed, "accuracy": acc})
        print(f"  epochs={epochs:3d} seed={seed} acc={acc:.3f}")
results["layers_epoch_scaling"] = exp3

results["meta"] = {"wall_time_s": time.time() - t0, "pre_fix_rev": CHUNK3_PRE_FIX_REV}
print(f"\nTotal wall time: {results['meta']['wall_time_s']:.1f}s")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {OUT_PATH}")
