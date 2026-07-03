"""Tests for the v2ecoli WCM Parquet history loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pbg_pennylane_adversarial.dataset_transform.wcm_loader import (
    load_wcm_history,
    auto_detect_targets,
    build_transition_pairs,
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

    def test_sorted_by_time_when_no_global_time(self):
        # Real pulled v2ecoli output (comparison_10s_16g_v2_aws) has "time", not
        # "global_time" -- confirm the loader doesn't silently skip sorting.
        history = _hive_history({
            "generation=0/agent_id=00": {"time": [2.0, 0.0, 1.0], "x": [1.0, 2.0, 3.0]},
        })
        df = load_wcm_history(history)
        assert df["time"].to_list() == [0.0, 1.0, 2.0]

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


# ── build_transition_pairs ───────────────────────────────────────────────────


class TestBuildTransitionPairs:
    def _df(self):
        # Two trajectories (lineage_seed=0/gen=0 and lineage_seed=0/gen=1), 3 and 2
        # rows respectively, deliberately out of time order to check sort-before-pair.
        return pl.DataFrame({
            "lineage_seed": [0, 0, 0, 0, 0],
            "generation": [0, 0, 0, 1, 1],
            "agent_id": [0, 0, 0, 0, 0],
            "time": [2.0, 0.0, 1.0, 10.0, 11.0],
            "a": [20.0, 0.0, 10.0, 100.0, 110.0],
            "b": [200.0, 0.0, 100.0, 1000.0, 1100.0],
        })

    def test_shapes_and_no_cross_trajectory_leakage(self):
        X, Y, DT, traj_id = build_transition_pairs(self._df(), feature_cols=["a", "b"])
        # 3-row trajectory -> 2 pairs, 2-row trajectory -> 1 pair = 3 total.
        assert X.shape == (3, 2)
        assert Y.shape == (3, 2)
        assert DT.shape == (3,)
        assert traj_id.shape == (3,)
        assert len(set(traj_id.tolist())) == 2

    def test_pairs_are_correctly_ordered_by_time(self):
        X, Y, DT, traj_id = build_transition_pairs(self._df(), feature_cols=["a", "b"])
        # Within the first trajectory (sorted 0.0, 1.0, 2.0), pairs are (0->1), (1->2).
        first_traj = traj_id[0]
        mask = traj_id == first_traj
        assert np.allclose(X[mask], [[0.0, 0.0], [10.0, 100.0]])
        assert np.allclose(Y[mask], [[10.0, 100.0], [20.0, 200.0]])
        assert np.allclose(DT[mask], [1.0, 1.0])

    def test_second_trajectory_not_paired_with_first(self):
        X, Y, DT, traj_id = build_transition_pairs(self._df(), feature_cols=["a", "b"])
        second_traj = traj_id[-1]
        mask = traj_id == second_traj
        assert mask.sum() == 1
        assert np.allclose(X[mask], [[100.0, 1000.0]])
        assert np.allclose(Y[mask], [[110.0, 1100.0]])
        assert np.allclose(DT[mask], [1.0])

    def test_single_row_trajectory_produces_no_pairs(self):
        df = pl.DataFrame({
            "lineage_seed": [0],
            "generation": [0],
            "agent_id": [0],
            "time": [0.0],
            "a": [1.0],
        })
        X, Y, DT, traj_id = build_transition_pairs(df, feature_cols=["a"])
        assert X.shape == (0, 1)
        assert Y.shape == (0, 1)
        assert DT.shape == (0,)
        assert traj_id.shape == (0,)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            build_transition_pairs(self._df(), feature_cols=["not_a_column"])
