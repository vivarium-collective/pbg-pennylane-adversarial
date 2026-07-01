"""Transform an arbitrary DataFrame into PLAP-ready train/test splits."""

from __future__ import annotations

import polars as pl


def resolve_target_column(df: pl.DataFrame, target_col: str | None) -> str:
    """Determine which column is the target.

    Priority:
    1. Explicit ``target_col`` argument.
    2. A column literally named ``"target"``.
    3. The last column.
    """
    if target_col is not None:
        return target_col
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def validate_classification_target(series: pl.Series) -> None:
    """Ensure the target column is scalar and has at least 2 unique values."""
    if series.dtype in (pl.List, pl.Struct, pl.Array):
        raise ValueError(
            f"Target column '{series.name}' must be scalar "
            f"(got {series.dtype})."
        )
    n_unique = series.n_unique()
    if n_unique < 2:
        raise ValueError(
            f"Target column '{series.name}' has only {n_unique} unique "
            f"value(s) — need at least 2 for classification."
        )


def encode_labels(series: pl.Series) -> tuple[pl.Series, dict[int, str]]:
    """Label-encode a target series to contiguous 0..C-1.

    Handles both string and integer labels. Integer labels are remapped
    to contiguous 0..C-1 if they aren't already.

    Returns (encoded_series, {encoded_value: original_label_str}).
    """
    if series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        unique = sorted(series.unique().to_list())
        mapping = {i: str(v) for i, v in enumerate(unique)}
        value_to_idx = {v: i for i, v in enumerate(unique)}
        return series.replace_strict(value_to_idx).cast(pl.Int64), mapping
    else:
        cat = series.cast(pl.Categorical)
        physical = cat.to_physical().cast(pl.Int64)
        categories = cat.cat.get_categories().to_list()
        mapping = {i: str(categories[i]) for i in range(len(categories))}
        return physical, mapping


def normalize_features(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_cols: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float], dict[str, float]]:
    """Z-score normalize feature columns (fit on train, transform both).

    Returns (train_norm, test_norm, means, stds).
    """
    stats: list[tuple[str, float, float]] = []
    exprs: list[pl.Expr] = []
    for col in feature_cols:
        m = train[col].mean()
        s = train[col].std()
        if s == 0.0:
            s = 1.0
        stats.append((col, m, s))
        exprs.append(
            ((pl.col(col) - m) / s).alias(col)
        )
    means = {c: m for c, m, _ in stats}
    stds = {c: s for c, _, s in stats}
    return train.with_columns(exprs), test.with_columns(exprs), means, stds


def transform(
    df: pl.DataFrame,
    *,
    target_col: str | None = None,
    feature_cols: list[str] | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    normalize: bool = True,
) -> dict:
    """Convert a raw DataFrame into PLAP-ready train/test splits.

    Parameters
    ----------
    df : polars.DataFrame
        Raw dataset.
    target_col : str, optional
        Name of the target/label column. Auto-detected if ``None``.
    feature_cols : list[str], optional
        Columns to use as features. Auto-detected (all except target) if ``None``.
    train_ratio : float
        Proportion of data for training (default 0.8).
    seed : int
        Random seed for reproducible split.
    normalize : bool
        Whether to z-score normalize features (default True).

    Returns
    -------
    dict with keys:
        train_images, train_labels, test_images, test_labels,
        input_dim, output_dim, n_train, n_test, label_map,
        means (if normalize), stds (if normalize)
    """
    target_col = resolve_target_column(df, target_col)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {df.columns}"
        )

    target_series = df[target_col]
    validate_classification_target(target_series)

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    non_numeric = [
        c for c in feature_cols
        if df.schema[c] not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                pl.Float32, pl.Float64)
    ]
    if non_numeric:
        raise ValueError(
            f"Feature columns must be numeric. Non-numeric columns found: {non_numeric}. "
            f"Use --feature-cols to specify which columns are features."
        )

    null_mask = df.select(pl.any_horizontal(pl.all().is_null())).to_series()
    if null_mask.any():
        n_nulls = null_mask.sum()
        raise ValueError(
            f"Dataset contains {n_nulls} row(s) with null values. "
            f"Use --drop-nulls to drop them automatically."
        )

    # Label encode target
    labels_encoded, label_map = encode_labels(target_series)
    df = df.with_columns(labels_encoded.alias("__label__"))

    # Stratified train/test split
    df = df.with_row_index("__idx__")
    train_idx = (
        df
        .group_by("__label__")
        .agg(pl.col("__idx__").sample(fraction=train_ratio, seed=seed))
        .explode("__idx__", empty_as_null=False)
        .get_column("__idx__")
    )
    train_idx_set = set(train_idx.to_list())
    train_df = df.filter(pl.col("__idx__").is_in(train_idx_set)).drop("__idx__")
    test_df = df.filter(~pl.col("__idx__").is_in(train_idx_set)).drop("__idx__")

    train_images = train_df.select(feature_cols)
    test_images = test_df.select(feature_cols)

    if normalize:
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        train_images, test_images, means, stds = normalize_features(
            train_images, test_images, feature_cols
        )

    result = {
        "train_images": train_images.to_numpy().tolist(),
        "train_labels": train_df["__label__"].to_list(),
        "test_images": test_images.to_numpy().tolist(),
        "test_labels": test_df["__label__"].to_list(),
        "input_dim": len(feature_cols),
        "output_dim": len(label_map),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "label_map": label_map,
    }

    if normalize:
        result["means"] = means
        result["stds"] = stds

    return result
