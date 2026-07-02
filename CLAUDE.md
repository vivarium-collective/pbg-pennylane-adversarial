# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph) wrapper around the PennyLane
["adversarial attacks on QML classifiers"](https://pennylane.ai/demos/tutorial_adversarial_attacks_QML) tutorial.
It packages the train → PGD attack → adversarial-retrain → evaluate pipeline as a PBG `Process` plus composite
generators, and adds a dataset-formatting CLI so **any** classification dataset (not just the tutorial's
PlusMinus digits) can be run through it. It is also a `pbg-superpowers` **workspace** (see `workspace.yaml`,
`AGENTS.md`) used for biology-adjacent investigations (e.g. classifying vEcoli/v2ecoli whole-cell-model
trajectories) — those live under `workspace/investigations/` and `workspace/studies/`, separate from the
installable package.

## Setup & commands

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

```bash
pytest                                    # full suite
pytest tests/test_processes.py -k full_pipeline_synthetic   # single test
pytest -m "not timeout"                   # most pipeline tests are @pytest.mark.timeout(300); they run real PennyLane/torch training and are slow
```

CLI (installed as `adversarial`, entry point `app.cli:main`):

```bash
adversarial datasets format <file.csv> [--target-col ...] [--output-format h5|json|parquet]
adversarial pipeline run <formatted-dataset> [--num-qubits N] [--output results]
```

Demo report (standalone HTML, not the CLI's `pipeline run --output`):

```bash
python demo/demo_report.py
```

## Architecture

**Data flow is the core design constraint**: `PennyLaneAdversarialProcess` (`pbg_pennylane_adversarial/processes.py`)
takes data through four PBG input ports — `train_images`, `train_labels`, `test_images`, `test_labels` — rather
than being hardcoded to one dataset. `input_dim`/`output_dim` are auto-detected from the data on first
`update()` (shape[1] and number of unique labels) unless overridden in config. If no data arrives via the ports,
it falls back to loading PennyLane's built-in PlusMinus dataset (backward-compat default). Data is cached
internally after the first step, so callers don't need to keep it in the state dict.

**The process is a phase state machine**, not a single-shot computation. Each `update()` call advances one epoch
or one evaluation phase and returns the next phase name in its output; callers loop `update()` until
`phase == "done"`:

```
init → training (N epochs) → benign_eval → attack (PGD) → adversarial_training (N epochs) → done
```

Model architecture is fixed regardless of data: `StronglyEntanglingLayers` circuit on `lightning.qubit`, wrapped
in a `qml.qnn.TorchLayer` inside a small `torch.nn.Module`, trained with `CrossEntropyLoss`/Adam. `num_qubits`/
`num_layers` are configurable; the circuit structure is not. A **data-reuploading multiplier** (`num_reup`) is
computed in `_build_model` to satisfy `n_layers * n_qubits * 3 == num_reup * input_dim` — this is derived, never
set directly. If `output_dim > num_qubits`, `num_qubits` is silently raised to match (with a warning).

**Two ways to run the pipeline**, both converging on the same `Process`:
1. Directly: instantiate `PennyLaneAdversarialProcess`, set `train_images`/etc. on the state dict, loop `update()`.
2. Via composites (`pbg_pennylane_adversarial/composites/adversarial.py`): `_build_adversarial_document()` builds
   a full PBG document (process + stores + `RAMEmitter`) with data embedded in stores; wrap in
   `Composite({"state": doc}, core=core)` and call `sim.run(...)`. Three `@composite_generator`-registered
   presets (`adversarial_baseline`, `adversarial_robust`, `adversarial_lightweight`) and one that loads a
   formatted dataset from disk (`adversarial_from_formatted`) wrap this builder with different hyperparameters.
   Composite generators self-register with `pbg_superpowers.composite_generator._REGISTRY` on import — no
   manual registration needed, and likewise `PennyLaneAdversarialProcess` registers via
   `bigraph_schema.package.discover` once the package is installed (no manual `register_link()` calls needed
   for the process either, only for third-party pieces like `RAMEmitter`).

**Dataset formatting is a separate pipeline stage** (`pbg_pennylane_adversarial/dataset_transform/`), decoupled
from the Process itself: `reader.py` auto-detects format from extension (CSV/TSV/Parquet/JSON/Excel/Feather/
HDF5/npy/npz/pickle) and loads into a polars DataFrame; `transform.py` resolves the target column, label-encodes
it to contiguous `0..C-1`, does a stratified train/test split, and optionally z-score normalizes features (fit
on train, applied to both); `writer.py` persists the result as h5/json/parquet; `loader.py` (`load_formatted`)
reads a written artifact back into the dict shape the Process/composites expect. The CLI's `datasets format`
command and `adversarial_from_formatted`/`load_formatted` in the composites module are the two integration
points between this stage and the pipeline — going through the CLI-produced artifact is the intended path for
real (non-synthetic) datasets, rather than hand-building the four arrays.

**`app/cli.py`** is a `typer` app with two sub-apps: `datasets` (currently just `format`) and `pipeline` (`run`).
`pipeline run` loads a formatted artifact, drives the same `update()` loop as above for up to 200 steps, prints
progress/summary, and optionally writes a JSON results file plus an HTML report via
`pbg_pennylane_adversarial/report.py` (`generate_run_report`, built with Plotly — no JS framework).

## Conventions from AGENTS.md (framework-wide, applies to all pbg-superpowers workspaces)

- **Feature/fix PRs** ship reusable infrastructure (new Processes, schema fields, scripts) against `main`:
  conventional-commit title, ready-for-review (not draft).
- **Investigation PRs** are long-running, non-mergeable branches for a research question (study YAMLs, reports,
  references). Title prefixed `investigation:`, opened as `--draft`, body must state it's not a merge target and
  list which feature PRs it depends on.
- When in doubt, put reusable code in a feature PR against `main` rather than on an investigation branch.
- Closing an investigation is done via `/pbg-investigation close <slug>`, not by hand.

## Known gaps (see NEXT_STEPS.md for full detail)

- No `datasets from-wcm` subcommand yet for pulling directly from v2ecoli/vEcoli Parquet simulation output —
  currently requires manual CSV curation first.
- No classical-baseline (logistic regression / random forest) comparison in the pipeline or report.