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
    if "global_time" in df.columns:
        df = df.sort("global_time")
    return df


def auto_detect_targets(df: pl.DataFrame) -> list[str]:
    """Suggest cell-cycle-phase target column candidates.

    ``listeners__replication_data__*`` columns track origin/terminus counts
    and replication state — natural classification targets for cell-cycle
    phase.
    """
    return sorted(c for c in df.columns if c.startswith("listeners__replication_data__"))
