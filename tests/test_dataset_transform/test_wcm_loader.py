"""Tests for the v2ecoli WCM Parquet history loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from pbg_pennylane_adversarial.dataset_transform.wcm_loader import (
    load_wcm_history,
    auto_detect_targets,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _hive_history(rows_per_partition: dict[str, dict]) -> Path:
    """Build a hive-partitioned history/ tree.

    ``rows_per_partition`` maps a hive path suffix (e.g.
    ``"generation=0/agent_id=00"``) to a dict of column -> list-of-values
    to write as one .pq file under that partition.
    """
    root = Path(tempfile.mkdtemp()) / "history"
    for i, (suffix, cols) in enumerate(rows_per_partition.items()):
        d = root / f"experiment_id=exp1/variant=0/lineage_seed=0/{suffix}"
        d.mkdir(parents=True)
        pl.DataFrame(cols).write_parquet(d / f"{100 * (i + 1)}.pq")
    return root


# ── load_wcm_history ─────────────────────────────────────────────────────────


class TestLoadWcmHistory:
    def test_flattens_agents_and_generations(self):
        history = _hive_history({
            "generation=0/agent_id=00": {
                "global_time": [0.0, 1.0],
                "listeners__mass__cell_mass": [500.0, 510.0],
            },
            "generation=0/agent_id=01": {
                "global_time": [0.0, 1.0],
                "listeners__mass__cell_mass": [500.0, 505.0],
            },
        })
        df = load_wcm_history(history)
        assert len(df) == 4
        assert "listeners__mass__cell_mass" in df.columns

    def test_recovers_hive_partition_columns(self):
        history = _hive_history({
            "generation=0/agent_id=00": {"global_time": [0.0], "x": [1.0]},
            "generation=1/agent_id=00": {"global_time": [1.0], "x": [2.0]},
        })
        df = load_wcm_history(history)
        assert set(df["generation"].to_list()) == {0, 1}
        assert set(df["agent_id"].to_list()) == {0}

    def test_sorted_by_global_time(self):
        history = _hive_history({
            "generation=0/agent_id=00": {"global_time": [2.0, 0.0, 1.0], "x": [1.0, 2.0, 3.0]},
        })
        df = load_wcm_history(history)
        assert df["global_time"].to_list() == [0.0, 1.0, 2.0]

    def test_tolerates_schema_drift(self):
        history = _hive_history({
            "generation=0/agent_id=00": {"global_time": [0.0], "a": [1.0], "b": [2.0]},
            "generation=0/agent_id=01": {"global_time": [0.0], "a": [1.0]},
        })
        df = load_wcm_history(history)
        assert len(df) == 2
        assert df["b"].null_count() == 1

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError, match="no history directory"):
            load_wcm_history("/tmp/nonexistent_wcm_history_12345")

    def test_empty_dir_raises(self):
        empty = Path(tempfile.mkdtemp())
        with pytest.raises(FileNotFoundError, match="no .pq history files"):
            load_wcm_history(empty)


# ── auto_detect_targets ──────────────────────────────────────────────────────


class TestAutoDetectTargets:
    def test_finds_replication_columns(self):
        df = pl.DataFrame({
            "listeners__replication_data__number_of_oric": [1, 2],
            "listeners__replication_data__critical_mass_per_oric": [1.0, 1.1],
            "listeners__mass__cell_mass": [500.0, 510.0],
        })
        candidates = auto_detect_targets(df)
        assert candidates == [
            "listeners__replication_data__critical_mass_per_oric",
            "listeners__replication_data__number_of_oric",
        ]

    def test_no_matches_returns_empty(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        assert auto_detect_targets(df) == []
