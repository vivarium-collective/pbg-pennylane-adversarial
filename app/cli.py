"""adversarial CLI — format arbitrary classification datasets for PLAP."""

from __future__ import annotations

from pathlib import Path

import typer

from pbg_pennylane_adversarial.dataset_transform.reader import (
    SUPPORTED_EXTENSIONS,
)
from pbg_pennylane_adversarial.dataset_transform.transform import transform
from pbg_pennylane_adversarial.dataset_transform.writer import write, OutputFormat

datasets_app = typer.Typer()
cli = typer.Typer()
cli.add_typer(datasets_app, name="datasets")


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


def main():
    cli()


if __name__ == "__main__":
    main()
