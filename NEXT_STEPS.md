# NEXT STEPS — Implementation Plan

All gaps sorted by Impact × Effort × Severity magnitude (descending).

---

## 1. WCM Extraction Tool (`I=3, E=3, S=2 → 18`)

**Gap**: No way to go from v2ecoli simulation outputs → formatted dataset without manual CSV curation.

**Source**: v2ecoli at `~/vivarium-app/v2ecoli`. Outputs are Parquet hive dirs:
```
out/workflow/parquet/<experiment_id>/history/
  experiment_id=<id>/variant=<v>/lineage_seed=<s>/generation=<g>/agent_id=<id>/N.pq
```

**Pre-req — parca cache**: runs need `out/cache/sim_data_cache.dill`. If missing or stale, run:
```
cd ~/vivarium-app/v2ecoli
uv run v2ecoli-parca --mode fast --cache-dir out/cache  # ~30 min
```
The cache at `out/cache/` exists and looks valid (has `sim_data_cache.dill`), but verify before running a full workflow.

**Plan**:
1. Add `adversarial datasets from-wcm` subcommand to `app/cli.py`:
   - Walks the Parquet hive directory tree
   - Scans all agents/generations/time steps
   - Flattens into rows × columns (one row per agent per time step)
   - Accepts `--feature-cols`, `--target-col`, `--train-ratio`, `--seed`, `--normalize`, `--output-format`, `--output`
   - Calls existing `transform()` then `write()` for downstream pipeline compatibility
2. Add a helper function in `pbg_pennylane_adversarial/dataset_transform/wcm_loader.py`:
   - `load_wcm_history(history_dir)` → `pl.DataFrame` of all time steps (reuses pattern from `v2ecoli.library.parquet_viz.load_run_history`)
   - `auto_detect_targets(df)` → suggests cell-cycle phase candidates from `listeners__replication_data__*` columns
3. Verify: `uv run adversarial datasets from-wcm --help` → sensible opts

---

## 2. Classical Baseline Comparison in Report (`I=3, E=1, S=2 → 6`)

**Gap**: Report shows quantum metrics but no reference — can't tell if the QML adds value.

**Plan**:
1. Add `--baselines` flag to `pipeline run` (optional, `default=False`):
   - When set, train `LogisticRegression(max_iter=1000)` and `RandomForestClassifier(n_estimators=100)` on same train/test split
   - Also run PGD perturbation on classical models (project perturbed features back to valid range after each step — note PGD is L_infinity, works on any differentiable model; for sklearn we approximate by measuring accuracy drop on perturbed test set after one-shot perturbation)
2. Extend `_save_outputs` / `records` to include `baselines: {logistic_regression: {benign_accuracy, adversarial_accuracy}, random_forest: {benign_accuracy, adversarial_accuracy}}`
3. Extend `generate_run_report()` in `report.py` to render a **Baseline Comparison** section:
   - Grouped bar chart: benign accuracy × adversarial accuracy for QML + each baseline
   - Metrics card per baseline
   - If quantum model is within 5 pp of best classical, call it "competitive"; otherwise note the gap
4. Add sklearn to `pyproject.toml` deps (if not already; uv picks it up transitively from v2ecoli)

**Status: done.** `pbg_pennylane_adversarial/baselines.py` (`run_baselines`), `--baselines` flag in
`pipeline run`, Baseline Comparison section in `report.py`. Also added a **transfer-attack** number
per baseline (own-attack robustness vs. accuracy under the QML PGD attack's exact perturbation replayed
against the baseline) — the QML process now exposes `perturbation_delta` as an output port
(`processes.py`) so the CLI can pass it to `run_baselines(..., transfer_delta=...)`. This was a deliberate
scope addition beyond the original plan text, agreed with the user in-session.

---

## 3. High-Dimensional Feature Scaling (`I=1, E=2, S=1 → 2`)

**Gap**: `num_reup` dilutes per-feature expressivity when `input_dim` is large relative to `num_qubits * num_layers * 3`.

**Plan** (deferred — only act if >50 features hurt accuracy):
1. Check: does adding more features monotonically decrease accuracy on a fixed number of qubits/layers? If yes, consider:
   - Auto-raise `num_layers` based on `input_dim` (e.g., `num_layers = max(config.num_layers, ceil(input_dim / num_qubits))`)
   - Or add a PCA step before the circuit: `input_dim → min(input_dim, num_qubits * num_layers)` via `sklearn.decomposition.PCA`
2. Log a warning in `_build_model` if `num_reup > 3` to alert the user.

**Status: done, scope-adjusted from the plan based on empirical testing.**
- **Trigger confirmed**: on a fixed small circuit (`num_qubits=4, num_layers=4` → `weights_elements=48`),
  benign accuracy collapses to ~chance once `input_dim` approaches/exceeds `weights_elements`
  (`num_reup` hitting 1), regardless of which feature column carries the signal — not just a truncation
  artifact of `forward()`'s `repeated[:weights_elements]`.
- **Auto-raising `num_layers` (the plan's first option) was tried and empirically rejected**: even after
  raising `num_layers` to restore `num_reup >= 2` and training for up to 50 epochs, accuracy stayed at
  chance. A deeper `StronglyEntanglingLayers` circuit is measurably harder to train (consistent with
  barren-plateau-type effects), so growing the circuit is not a safe default fix. Instead, the implemented
  option is a PCA step (`sklearn.decomposition.PCA`, fit on training data only), which keeps
  `num_qubits`/`num_layers` fixed and reduces `input_dim` to `weights_elements // MIN_NUM_REUP`
  (`MIN_NUM_REUP = 2`, a class constant on `PennyLaneAdversarialProcess`) whenever the raw `input_dim`
  would exceed it. `n_components` is additionally clamped to `n_train_samples` to satisfy sklearn's PCA
  constraint on small datasets. A `UserWarning` is emitted whenever this triggers.
- **Honest caveat**: the PCA fix measurably helps (mean accuracy improves vs. doing nothing) but does
  *not* reliably restore accuracy to the same level as native low-dimensional data in all cases — this
  remains a genuinely hard optimization problem (small-sample noise, limited training budget, circuit
  expressivity) beyond this chunk's minimal-effort scope. Treat this as harm reduction, not a full fix.
- The plan's second bullet ("warn if `num_reup > 3`") was inverted based on the data: the actual risk
  zone is `num_reup` being *too low* (≤1), not too high, so the warning fires on the low-redundancy
  condition instead (see the PCA-trigger warning above, which subsumes it).
- Tests: `tests/test_processes.py::test_high_input_dim_triggers_pca_reduction`,
  `::test_low_input_dim_no_pca_warning`. Full non-timeout suite: 61 passed (was 59; two new tests added).
  Full suite (incl. `@pytest.mark.timeout` real-training tests): 65 passed, 4 failed — the 4 failures
  (`test_composite_assembly_with_data`, `test_composite_assembly_plusminus`,
  `test_adversarial_baseline_generator`, `test_adversarial_lightweight_generator`) are a pre-existing
  link-registration ordering issue, confirmed present on the unmodified `main` baseline before this
  chunk's changes — unrelated to this work, not fixed (out of scope).

---

## 4. A1 Baseline Decision Gate — persistence, DMD, SINDy on real WCM transition data (`I=3, E=1, S=2 → 6`)

**Gap**: `todo.md`'s A1 plan (QGRNN-style graph-structured surrogate) has its own phase-4 decision
gate — "if MLP/GNN/QGRNN don't clearly beat persistence and DMD, stop, the ablation is measuring
nothing" — but that gate has never been run against real data. Phases 0–2 (literature check, data
spike, circuit spike) and phase 3 (`QGRNNSurrogate`/`ClassicalGNNSurrogate` model classes) are
already done and committed, but no model has been trained on real transition data yet, and no
baseline has been computed against it either.

**Source**: `todo.md` §2 (A1), phases 1–4;
`pbg_pennylane_adversarial/dataset_transform/wcm_loader.py` (`load_wcm_history`,
`build_transition_pairs`, already built + tested);
`pbg_pennylane_adversarial/qgrnn_surrogate.py` (`QGRNNSurrogate`, `ClassicalGNNSurrogate`,
already built + tested, not yet trained on real data). Confirmed prior-session decisions: a
60-second prediction stride (native 1-second resolution is dominated by trivial persistence);
N=4 real trajectories (4 lineage seeds × 3 generations pulled from `comparison_10s_16g_v2_aws`,
not the 12 `todo.md` originally assumed).

**Plan**:
1. Build the real transition dataset: `load_wcm_history()` on the 8 pulled parquet shards →
   `build_transition_pairs()` at the confirmed 60-second stride, restricted to the 8 real
   mass+chromosome columns (`cell_mass`, `dry_mass`, `protein_mass`, `rna_mass`, `dna_mass`,
   `volume`, `instantaneous_growth_rate`, `number_of_oric`), N=4 trajectories with one held out
   for test (mirroring `evaluate_surrogate.py`'s held-out-trajectory protocol).
2. Naive persistence baseline (`Ŷ = X`, no training).
3. Hand-rolled linear DMD baseline (rank-truncated SVD operator fit, per the numerical-stability
   fix already found during the phase-1 data spike — do not add `pydmd` as a dependency, per the
   earlier design-flaw finding: its API expects one sequential snapshot matrix, not pre-paired
   multi-trajectory `(X, Y)` pairs).
4. **New arm, not in `todo.md`'s original plan**: a SINDy-style sparse-regression baseline
   (Brunton, Proctor & Kutz 2016) — fit a sparse linear combination over a small library of
   candidate nonlinear terms per node. Needs less data than DMD+GNN and competes directly with
   A1's own "discovered sparse structure" claim more sharply than the classical-GNN arm does; if
   SINDy alone recovers the known couplings from 3 training trajectories, that's a sharper
   negative result for A1 than anything currently in the ablation.
5. Report per-node R²/RMSE for all three arms on the held-out trajectory; DMD's eigenvalue
   spectrum (slow/persistent-mode check, also a quantitative version of the growth-rate-variance
   check from anticipated criticism #2); SINDy's recovered sparse terms per node.
6. **Decision gate**: only proceed to training `QGRNNSurrogate`/`ClassicalGNNSurrogate` (both
   already built in `pbg_pennylane_adversarial/qgrnn_surrogate.py`, not yet trained on real data)
   once this step shows real, non-trivial dynamics left to model — i.e., don't spend the
   multi-session QGRNN/GNN training effort if persistence/DMD/SINDy already explain most of the
   achievable variance at N=4. If they do, report that plainly as its own finding rather than
   downplaying it to protect the neural/quantum narrative.

**Explicitly out of scope for this gap** (separate, not-yet-resolved items, tracked in project
memory rather than here): the `todo.md` "scrambled-graph control" is vacuous as currently written
(the graph starts fully-connected, so there's no alternate same-density topology to scramble
into) — a permutation-test-based fix has been proposed but not implemented, and only matters once
a trained coupling matrix exists to test, i.e. after this gap's decision gate passes. The
interventional/causal validation step (rerunning the `v2ecoli` composite with a perturbed
parameter) is also unaffected by this gap either way.

**Status: not started, queued next, awaiting go-ahead.**

---

## Running It End-to-End (Checklist)

```bash
# 0. Verify parca cache
ls ~/vivarium-app/v2ecoli/out/cache/sim_data_cache.dill

# 1. (if needed) Rebuild parca cache
cd ~/vivarium-app/v2ecoli && uv run v2ecoli-parca --mode fast

# 2. Run v2ecoli workflow
cd ~/vivarium-app/v2ecoli && uv run v2ecoli-workflow \
  --config v2ecoli/configs/two_generations.json \
  --out out/two_gen

# 3. Extract formatted dataset (once from-wcm is built)
cd ~/vivarium-app/pbg-pennylane-adversarial
uv run adversarial datasets from-wcm \
  --history-dir ~/vivarium-app/v2ecoli/out/two_gen/parquet/two_generations/history \
  --feature-cols listeners__mass__cell_mass,listeners__mass__instantaneous_growth_rate \
  --target-col listeners__replication_data__number_of_oric \
  --output wcm_formatted.h5

# 4. Run pipeline with baselines
uv run adversarial pipeline run wcm_formatted.h5 \
  --num-qubits 4 --output wcm_results --baselines
```

---

## Current State (as of this writing)

- **Fixed**: `num_qubits` auto-raised to match `output_dim` in `processes.py` (+warning)
- **Existing**: `datasets format` (CSV/TSV/Parquet/JSON/H5/Excel/etc → formatted artifact)
- **Existing**: `datasets from-wcm` (v2ecoli Parquet history hive dir → formatted artifact; gap 1 done)
- **Existing**: `pipeline run` (train → PGD → adversarial retrain → evaluate)
- **Existing**: HTML report with accuracy/loss charts, metrics cards, dataset/config badges,
  and an optional Baseline Comparison section (gap 2 done — `pipeline run --baselines`)
- **Existing**: high-`input_dim` features auto-reduced via PCA (fit on train only) to preserve
  data-reuploading redundancy, with a warning (gap 3 done — see caveat in gap 3's section above)
