"""Load v2ecoli/vEcoli whole-cell-model Parquet history output for PLAP.

v2ecoli workflow runs write one row per agent per time step under a hive-
partitioned Parquet tree:

    <history_dir>/experiment_id=<id>/variant=<v>/lineage_seed=<s>/generation=<g>/agent_id=<id>/N.pq

This module flattens that tree into a single polars DataFrame (one row per
agent-timestep, hive partition keys recovered as columns) suitable for
``dataset_transform.transform.transform()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def load_wcm_history(history_dir: str | Path) -> pl.DataFrame:
    """Load all `.pq` files under a v2ecoli Parquet `history/` directory.

    Reuses the pattern from ``v2ecoli.library.parquet_viz.load_run_history``,
    generalized to a run's whole hive tree (all variants/seeds/generations/
    agents) rather than a single run directory. ``missing_columns="insert"``
    tolerates schema drift across batches (a listener column may come and go
    as the cell-state shape changes tick to tick), filling absent columns
    with nulls rather than raising.
    """
    history_root = Path(history_dir)
    if not history_root.exists():
        raise FileNotFoundError(f"no history directory at {history_root}")

    pq_files = sorted(history_root.rglob("*.pq"))
    if not pq_files:
        raise FileNotFoundError(f"no .pq history files under {history_root}")

    df = pl.read_parquet(pq_files, hive_partitioning=True, missing_columns="insert")
    # Real v2ecoli Parquet output has been observed under both names -- "global_time"
    # per v2ecoli.library.parquet_viz's own convention, but "time" in at least one
    # real pulled experiment (comparison_10s_16g_v2_aws). Prefer "global_time" if
    # present, fall back to "time" rather than silently skipping the sort.
    time_col = "global_time" if "global_time" in df.columns else "time" if "time" in df.columns else None
    if time_col is not None:
        df = df.sort(time_col)
    return df


def build_transition_pairs(
    df: pl.DataFrame,
    feature_cols: list[str],
    group_keys: tuple[str, ...] = ("lineage_seed", "generation", "agent_id"),
    sort_col: str = "time",
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (X_t, Y_{t+1}) transition pairs from a flattened WCM history DataFrame.

    Groups by `group_keys` (one group per real trajectory), sorts each group by
    `sort_col`, and pairs each row's `feature_cols` values with the next row's
    within the same group -- never across a group boundary. Returns arrays whose
    field shapes match `pbg_torch.TransitionDataset` (`X`, `Y`, `DT`, `traj_id`):

        X: (n, len(feature_cols))   feature values at t
        Y: (n, len(feature_cols))   feature values at t+1
        DT: (n,)                    sort_col delta between t and t+1
        traj_id: (n,)               integer id of the source trajectory (0..n_traj-1)

    `DT` is not cosmetic: real WCM Parquet output is written in per-generation
    time-chunked shards, so a naive "sort then pair" can silently splice a large,
    non-physical time gap in at a shard boundary as if it were a normal step.
    Callers should inspect/filter on `DT` (e.g. drop pairs whose DT deviates from
    the expected raster interval) rather than assume every returned pair is a
    normal single-step transition.

    `stride` subsamples each group (every `stride`-th row, by `sort_col` order)
    before pairing consecutive (now-subsampled) rows -- e.g. `stride=60` on
    native 1-second-resolution real WCM data produces ~60-second-interval
    transition pairs instead of trivial 1-second ones (confirmed prior finding:
    native resolution is dominated by trivial persistence). Default `stride=1`
    reproduces the original single-step-pairing behavior exactly.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    missing = [c for c in (*group_keys, sort_col, *feature_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found in df: {missing}")

    ordered = df.select([*group_keys, sort_col, *feature_cols]).sort([*group_keys, sort_col])

    xs, ys, dts, tids = [], [], [], []
    for traj_id, (_, group) in enumerate(ordered.group_by(group_keys, maintain_order=True)):
        if stride > 1:
            group = group[::stride]
        if len(group) < 2:
            continue
        feats = group.select(feature_cols).to_numpy()
        times = group[sort_col].to_numpy()
        xs.append(feats[:-1])
        ys.append(feats[1:])
        dts.append(times[1:] - times[:-1])
        tids.append(np.full(len(group) - 1, traj_id, dtype=np.int64))

    if not xs:
        n_feat = len(feature_cols)
        return (
            np.empty((0, n_feat)),
            np.empty((0, n_feat)),
            np.empty((0,)),
            np.empty((0,), dtype=np.int64),
        )

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    DT = np.concatenate(dts, axis=0)
    traj_id_arr = np.concatenate(tids, axis=0)
    return X, Y, DT, traj_id_arr


def auto_detect_targets(df: pl.DataFrame) -> list[str]:
    """Suggest cell-cycle-phase target column candidates.

    ``listeners__replication_data__*`` columns track origin/terminus counts
    and replication state — natural classification targets for cell-cycle
    phase.
    """
    return sorted(c for c in df.columns if c.startswith("listeners__replication_data__"))
