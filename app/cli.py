"""adversarial CLI — format datasets and run the adversarial pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from pbg_pennylane_adversarial.dataset_transform.reader import (
    SUPPORTED_EXTENSIONS,
)
from pbg_pennylane_adversarial.dataset_transform.transform import transform
from pbg_pennylane_adversarial.dataset_transform.writer import write, OutputFormat
from pbg_pennylane_adversarial.utils import run_subprocess

datasets_app = typer.Typer()
pipeline_app = typer.Typer()
demo_app = typer.Typer()
cli = typer.Typer()
cli.add_typer(datasets_app, name="datasets")
cli.add_typer(pipeline_app, name="pipeline")
cli.add_typer(demo_app, name="demo")


# ---------------------------------------------------------------------------
# notebook demo
# ---------------------------------------------------------------------------
@demo_app.command(
    name="notebook",
)
def demo_notebook(module: str | None):
    run_subprocess(cmd=f"uv run marimo edit {module or 'sandbox'}.py")


# ---------------------------------------------------------------------------
# datasets format (existing)
# ---------------------------------------------------------------------------


@datasets_app.command(
    name="format",
    help="Format any classification dataset for PLAP input.",
)
def format_dataset(
    filepath: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to the raw dataset file.",
    ),
    target_col: str | None = typer.Option(
        None, "--target-col", "-t",
        help="Name of the target/label column. Auto-detected if omitted "
             "(searches for a 'target' column, then falls back to last column).",
    ),
    feature_cols: list[str] | None = typer.Option(
        None, "--feature-cols", "-f",
        help="Comma-separated feature column names. All non-target columns "
             "used if omitted.",
    ),
    train_ratio: float = typer.Option(
        0.8, "--train-ratio",
        min=0.1, max=0.99,
        help="Fraction of data for training.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for reproducible train/test split.",
    ),
    no_normalize: bool = typer.Option(
        False, "--no-normalize",
        help="Skip z-score normalization of features.",
    ),
    drop_nulls: bool = typer.Option(
        False, "--drop-nulls",
        help="Silently drop rows with null values instead of erroring.",
    ),
    output_format: OutputFormat = typer.Option(
        "h5", "--output-format",
        help="Output format for the formatted dataset.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output file path (extension auto-set based on --output-format).",
    ),
):
    supported = ", ".join(SUPPORTED_EXTENSIONS)
    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        typer.echo(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {supported}",
            err=True,
        )
        raise typer.Exit(1)

    from pbg_pennylane_adversarial.dataset_transform.reader import read_dataset

    typer.echo(f"Reading {filepath}...")
    try:
        df = read_dataset(filepath)
    except Exception as e:
        typer.echo(f"Error reading dataset: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(df)} rows × {len(df.columns)} columns: {df.columns}")

    if drop_nulls:
        before = len(df)
        df = df.drop_nulls()
        dropped = before - len(df)
        if dropped:
            typer.echo(f"Dropped {dropped} row(s) with null values.")

    typer.echo(f"Transforming (target={target_col or '<auto>'}, "
               f"train_ratio={train_ratio}, normalize={not no_normalize})...")
    try:
        result = transform(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            train_ratio=train_ratio,
            seed=seed,
            normalize=not no_normalize,
        )
    except Exception as e:
        typer.echo(f"Error transforming dataset: {e}", err=True)
        raise typer.Exit(1)

    if output is None:
        output = filepath.with_stem(filepath.stem + "_formatted")
    out_path = write(result, output, fmt=output_format)

    typer.echo(
        f"Done — {result['n_train']} train / {result['n_test']} test samples, "
        f"{result['input_dim']} features, {result['output_dim']} classes"
    )
    typer.echo(f"Written to: {out_path}")


# ---------------------------------------------------------------------------
# datasets from-wcm
# ---------------------------------------------------------------------------


@datasets_app.command(
    name="from-wcm",
    help="Extract a formatted PLAP dataset from v2ecoli/vEcoli Parquet history output.",
)
def from_wcm(
    history_dir: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to the v2ecoli workflow's Parquet 'history/' directory.",
    ),
    feature_cols: list[str] = typer.Option(
        ..., "--feature-cols", "-f",
        help="Feature column name(s), e.g. -f listeners__mass__cell_mass "
             "-f listeners__mass__instantaneous_growth_rate.",
    ),
    target_col: str | None = typer.Option(
        None, "--target-col", "-t",
        help="Name of the target/label column. If omitted, candidate "
             "cell-cycle-phase columns are printed and the command exits.",
    ),
    train_ratio: float = typer.Option(
        0.8, "--train-ratio",
        min=0.1, max=0.99,
        help="Fraction of data for training.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for reproducible train/test split.",
    ),
    no_normalize: bool = typer.Option(
        False, "--no-normalize",
        help="Skip z-score normalization of features.",
    ),
    drop_nulls: bool = typer.Option(
        False, "--drop-nulls",
        help="Silently drop rows with null values instead of erroring.",
    ),
    output_format: OutputFormat = typer.Option(
        "h5", "--output-format",
        help="Output format for the formatted dataset.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output file path (extension auto-set based on --output-format).",
    ),
):
    from pbg_pennylane_adversarial.dataset_transform.wcm_loader import (
        load_wcm_history,
        auto_detect_targets,
    )

    typer.echo(f"Scanning {history_dir}...")
    try:
        df = load_wcm_history(history_dir)
    except Exception as e:
        typer.echo(f"Error loading WCM history: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(df)} agent-timesteps × {len(df.columns)} columns.")

    if target_col is None:
        candidates = auto_detect_targets(df)
        typer.echo("No --target-col given. Candidate cell-cycle-phase columns:")
        for c in candidates:
            typer.echo(f"  {c}")
        if not candidates:
            typer.echo("  (none found — pass --target-col explicitly)")
        raise typer.Exit(1)

    if drop_nulls:
        before = len(df)
        df = df.drop_nulls()
        dropped = before - len(df)
        if dropped:
            typer.echo(f"Dropped {dropped} row(s) with null values.")

    typer.echo(f"Transforming (target={target_col}, features={feature_cols}, "
               f"train_ratio={train_ratio}, normalize={not no_normalize})...")
    try:
        result = transform(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            train_ratio=train_ratio,
            seed=seed,
            normalize=not no_normalize,
        )
    except Exception as e:
        typer.echo(f"Error transforming dataset: {e}", err=True)
        raise typer.Exit(1)

    if output is None:
        output = history_dir.parent / "wcm_formatted"
    out_path = write(result, output, fmt=output_format)

    typer.echo(
        f"Done — {result['n_train']} train / {result['n_test']} test samples, "
        f"{result['input_dim']} features, {result['output_dim']} classes"
    )
    typer.echo(f"Written to: {out_path}")


# ---------------------------------------------------------------------------
# pipeline run
# ---------------------------------------------------------------------------


@pipeline_app.command(
    name="run",
    help="Run the adversarial pipeline on a formatted dataset.",
)
def run_pipeline(
    dataset_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Path to the formatted dataset (.h5 / .json / Parquet directory).",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Output stem for JSON results + HTML report "
             "(e.g. --output results → results.json + results.html).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress per-step progress output.",
    ),
    # circuit
    num_qubits: int = typer.Option(
        4, "--num-qubits", "-n",
        help="Number of qubits in the quantum circuit.",
    ),
    num_layers: int = typer.Option(
        8, "--num-layers", "-l",
        help="Number of StronglyEntanglingLayers.",
    ),
    # training
    training_epochs: int = typer.Option(
        6, "--training-epochs",
        help="Number of initial training epochs.",
    ),
    batch_size: int = typer.Option(
        10, "--batch-size",
        help="Batch size for training.",
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed.",
    ),
    learning_rate: float = typer.Option(
        0.05, "--learning-rate",
        help="Adam learning rate.",
    ),
    # PGD attack
    epsilon: float = typer.Option(
        0.05, "--epsilon",
        help="PGD L_inf perturbation bound.",
    ),
    pgd_alpha: float = typer.Option(
        0.005, "--pgd-alpha",
        help="PGD step size.",
    ),
    pgd_iter: int = typer.Option(
        8, "--pgd-iter",
        help="Number of PGD iterations.",
    ),
    # adversarial retraining
    adversarial_epochs: int = typer.Option(
        3, "--adversarial-epochs",
        help="Number of adversarial retraining epochs.",
    ),
    baselines: bool = typer.Option(
        False, "--baselines",
        help="Also train classical LogisticRegression/RandomForest baselines "
             "on the same split and report benign/adversarial accuracy.",
    ),
):
    """Run the full train → PGD attack → adversarial retrain → evaluate
    pipeline on a pre-formatted dataset produced by ``adversarial datasets format``."""
    from pbg_pennylane_adversarial.dataset_transform.loader import load_formatted
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess
    from process_bigraph import allocate_core
    from process_bigraph.emitter import RAMEmitter

    # --- load ---
    if not quiet:
        typer.echo(f"Loading formatted dataset from {dataset_path}...")
    try:
        data = load_formatted(dataset_path)
    except Exception as e:
        typer.echo(f"Error loading formatted dataset: {e}", err=True)
        raise typer.Exit(1)

    n_train = data["n_train"]
    n_test = data["n_test"]
    input_dim = data["input_dim"]
    output_dim = data["output_dim"]

    if not quiet:
        typer.echo(
            f"  {n_train} train / {n_test} test, "
            f"{input_dim} features, {output_dim} classes"
        )

    # --- config ---
    config = {
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "training_epochs": training_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "pgd_alpha": pgd_alpha,
        "pgd_iter": pgd_iter,
        "adversarial_epochs": adversarial_epochs,
    }

    if not quiet:
        typer.echo(f"\nRunning PennyLane adversarial pipeline...")
        typer.echo(f"  {num_qubits} qubits, {num_layers} layers, "
                   f"lr={learning_rate}, batch={batch_size}")
        typer.echo(f"  {training_epochs} training epochs, "
                   f"epsilon={epsilon}, {pgd_iter} PGD iters")
        typer.echo(f"  {adversarial_epochs} adversarial epochs\n")

    # --- build core + process ---
    core = allocate_core()
    core.register_link("ram-emitter", RAMEmitter)

    proc = PennyLaneAdversarialProcess(config=config, core=core)
    state = proc.initial_state()

    state["train_images"] = data["train_images"]
    state["train_labels"] = data["train_labels"]
    state["test_images"] = data["test_images"]
    state["test_labels"] = data["test_labels"]

    # --- run loop ---
    records = []
    t0 = time.perf_counter()

    for step in range(200):
        result = proc.update(state, interval=1.0)
        state.update(result)
        records.append({
            "step": step,
            "phase": state["phase"],
            "epoch": state["epoch"],
            "loss": state["loss"],
            "accuracy": state["accuracy"],
            "benign_accuracy": state["benign_accuracy"],
            "adversarial_accuracy": state["adversarial_accuracy"],
            "robust_accuracy": state["robust_accuracy"],
            "adversarial_accuracy_drop": state["adversarial_accuracy_drop"],
            "n_queries": state["n_queries"],
        })
        if not quiet and state["phase"] != "done":
            _maybe_print_progress(step, state)
        if state["phase"] == "done":
            break

    wall_time = time.perf_counter() - t0

    if not quiet:
        _print_summary(records, wall_time)

    # --- classical baselines ---
    baseline_results = None
    if baselines:
        if not quiet:
            typer.echo("\nTraining classical baselines (LogisticRegression, RandomForest)...")
        try:
            from pbg_pennylane_adversarial.baselines import run_baselines
            baseline_results = run_baselines(
                data, epsilon=epsilon, seed=seed,
                transfer_delta=state.get("perturbation_delta"),
            )
        except ImportError:
            typer.echo("scikit-learn not installed — skipping --baselines.", err=True)
        else:
            if not quiet:
                _print_baselines(baseline_results)

    # --- save outputs ---
    if output is not None:
        _save_outputs(records, config, data, wall_time, dataset_path, output, quiet,
                      baselines=baseline_results)


def _maybe_print_progress(step, state):
    phase = state["phase"]
    epoch = state["epoch"]
    loss = state["loss"]
    acc = state.get("accuracy", 0.0)

    if phase == "training" and epoch > 0:
        line = f"  step {step:>3}  {phase}  epoch {epoch:>2}  loss={loss:.4f}  acc={acc:.1%}"
    elif phase == "adversarial_training" and epoch > 0:
        line = f"  step {step:>3}  {phase}  epoch {epoch:>2}  loss={loss:.4f}  acc={acc:.1%}"
    else:
        line = f"  step {step:>3}  {phase}"
    typer.echo(line)


def _print_summary(records, wall_time):
    last = records[-1] if records else {}

    phases = []
    for r in records:
        p = r.get("phase", "?")
        if not phases or phases[-1] != p:
            phases.append(p)

    benign = last.get("benign_accuracy", 0.0)
    adv = last.get("adversarial_accuracy", 0.0)
    robust = last.get("robust_accuracy", 0.0)
    drop = last.get("adversarial_accuracy_drop", 0.0)
    n_q = last.get("n_queries", 0)

    typer.echo("\n" + "=" * 48)
    typer.echo("Pipeline complete")
    typer.echo("=" * 48)
    typer.echo(f"  Phase progression:  {' → '.join(phases)}")
    typer.echo(f"  Steps:              {len(records)}")
    typer.echo(f"  Benign accuracy:    {benign:.1%}")
    typer.echo(f"  Adversarial acc.:   {adv:.1%}")
    typer.echo(f"  Robust accuracy:    {robust:.1%}")
    typer.echo(f"  Accuracy drop:      {drop:.1%}")
    typer.echo(f"  Circuit evals:      {n_q:,}")
    typer.echo(f"  Wall time:          {wall_time:.2f}s")
    typer.echo("=" * 48)


def _print_baselines(baseline_results):
    names = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest"}
    typer.echo("  " + "-" * 44)
    for key, metrics in baseline_results.items():
        line = (
            f"  {names.get(key, key):<20} benign={metrics['benign_accuracy']:.1%}  "
            f"adversarial={metrics['adversarial_accuracy']:.1%}"
        )
        if "transfer_adversarial_accuracy" in metrics:
            line += f"  transfer={metrics['transfer_adversarial_accuracy']:.1%}"
        typer.echo(line)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_outputs(records, config, data, wall_time, dataset_path, output, quiet,
                   baselines=None):
    stem = output.with_suffix("") if output.suffix else output
    json_path = stem.with_suffix(".json")
    html_path = stem.with_suffix(".html")

    dataset_info = {
        "source": str(dataset_path),
        "n_train": data["n_train"],
        "n_test": data["n_test"],
        "input_dim": data["input_dim"],
        "output_dim": data["output_dim"],
        "label_map": data.get("label_map", {}),
    }

    report = {
        "config": config,
        "dataset": dataset_info,
        "wall_time_s": round(wall_time, 3),
        "n_steps": len(records),
        "records": records,
        "baselines": baselines,
    }

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    if not quiet:
        typer.echo(f"\nResults written to: {json_path}")

    try:
        from pbg_pennylane_adversarial.report import generate_run_report

        html = generate_run_report(
            records=records,
            config=config,
            dataset_info=dataset_info,
            wall_time=wall_time,
            baselines=baselines,
        )
        with open(html_path, "w") as f:
            f.write(html)
        if not quiet:
            typer.echo(f"Report written to:  {html_path}")
    except ImportError:
        if not quiet:
            typer.echo("(HTML report skipped — report module not available)")


def main():
    cli()


if __name__ == "__main__":
    main()
