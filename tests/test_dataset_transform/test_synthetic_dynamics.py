"""Tests for the synthetic transition-pair generator (ported NonlinearProcessGenerator idea)."""

from __future__ import annotations

import numpy as np
import pytest

from pbg_pennylane_adversarial.dataset_transform.synthetic_dynamics import (
    SyntheticTransitionGenerator,
)


class TestGenerate:
    def test_output_shapes(self):
        gen = SyntheticTransitionGenerator(num_nodes=5, seed=0)
        data = gen.generate(num_trajectories=3, steps_per_trajectory=20, calibration_steps=50)
        assert data["X"].shape == (60, 5)
        assert data["Y"].shape == (60, 5)
        assert data["traj_id"].shape == (60,)
        assert len(set(data["traj_id"].tolist())) == 3

    def test_no_cross_trajectory_leakage(self):
        gen = SyntheticTransitionGenerator(num_nodes=2, seed=1)
        data = gen.generate(num_trajectories=2, steps_per_trajectory=10, calibration_steps=20)
        # each trajectory contributes exactly steps_per_trajectory pairs
        counts = np.bincount(data["traj_id"])
        assert list(counts) == [10, 10]

    def test_achievable_r2_present_for_every_node(self):
        gen = SyntheticTransitionGenerator(num_nodes=4, seed=2)
        data = gen.generate(num_trajectories=2, steps_per_trajectory=10, calibration_steps=500)
        assert set(data["achievable_r2"].keys()) == {"x0", "x1", "x2", "x3"}
        for r2 in data["achievable_r2"].values():
            assert 0.0 <= r2 <= 1.0

    def test_lower_process_noise_gives_higher_achievable_r2(self):
        low_noise = SyntheticTransitionGenerator(num_nodes=3, process_noise_std=0.01, seed=3)
        high_noise = SyntheticTransitionGenerator(num_nodes=3, process_noise_std=1.0, seed=3)
        low_data = low_noise.generate(num_trajectories=1, steps_per_trajectory=10, calibration_steps=2000)
        high_data = high_noise.generate(num_trajectories=1, steps_per_trajectory=10, calibration_steps=2000)
        low_mean_r2 = np.mean(list(low_data["achievable_r2"].values()))
        high_mean_r2 = np.mean(list(high_data["achievable_r2"].values()))
        assert low_mean_r2 > high_mean_r2

    def test_reproducible_with_same_seed(self):
        gen_a = SyntheticTransitionGenerator(num_nodes=3, seed=42)
        gen_b = SyntheticTransitionGenerator(num_nodes=3, seed=42)
        data_a = gen_a.generate(num_trajectories=2, steps_per_trajectory=15, calibration_steps=30)
        data_b = gen_b.generate(num_trajectories=2, steps_per_trajectory=15, calibration_steps=30)
        assert np.allclose(data_a["X"], data_b["X"])
        assert np.allclose(data_a["Y"], data_b["Y"])

    def test_zero_coupling_is_valid(self):
        gen = SyntheticTransitionGenerator(num_nodes=3, coupling_strength=0.0, seed=4)
        data = gen.generate(num_trajectories=1, steps_per_trajectory=10, calibration_steps=20)
        assert np.all(np.isfinite(data["X"]))

    def test_unknown_nonlinearity_raises(self):
        gen = SyntheticTransitionGenerator(nonlinearity="not_a_real_one", seed=5)
        with pytest.raises(ValueError):
            gen.generate(num_trajectories=1, steps_per_trajectory=5, calibration_steps=10)
