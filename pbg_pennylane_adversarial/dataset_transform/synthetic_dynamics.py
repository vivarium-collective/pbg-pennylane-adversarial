"""Controlled synthetic transition-pair generator with a *measured* achievable-R2 ceiling.

Ports the idea (not the code) of `pbg-pennylane-data-reuploading`'s
`NonlinearProcessGenerator` -- a parameterized nonlinear stochastic-process
generator whose whole point is a known-in-advance optimum, so a baseline
harness's result on real data can be checked against "does this harness even
detect signal when it's actually there" rather than trusted blind. See
`comparison.html` §6 (this repo's own comparative evaluation of the two
sister repos) for why this is worth porting: it is data-reuploading's one
component independently judged "genuinely novel" by that project's own
`.perspective.md` self-audit.

**Adaptation, not a copy**: the original generator targets *classification*
(binary label, "Bayes-optimal accuracy") because its host pipeline is a
classifier. This repo's live use for the concept is A1's phase-4 decision
gate, which is a *regression* problem (`X_t -> Y_{t+1}` transition
prediction, matching `wcm_loader.build_transition_pairs()`'s contract) --
so the analogous quantity here is an **achievable R2 ceiling** per output
node, not a classification accuracy.

**The original's own audit flaw, fixed here**: `NonlinearProcessGenerator`
hardcodes its "Bayes-optimal accuracy" as a constant (`1.0` or `0.85`)
rather than computing it, which its own `.perspective.md` flags as
"asserted, not empirically validated." This module instead *measures* the
ceiling directly: it runs the same noiseless drift function the returned
dataset's noisy `Y` was generated from, on a large calibration batch drawn
from the same trajectory distribution, and computes
`1 - Var(noise) / Var(Y)` empirically per node -- an honest measurement of
what the best possible predictor could achieve, not an asserted constant.

Usage as a positive control for A1's baseline gate: generate a synthetic
dataset shaped like the real 8-node WCM transition set, with a known,
measured achievable-R2 per node; run `dynamics_baselines.fit_dmd` /
`fit_sindy` against it exactly as against the real data. If a baseline
arm's measured R2 on synthetic data falls well short of the *measured*
achievable ceiling, that indicates a harness problem (not enough data, a
badly-chosen library/rank) rather than evidence about the real WCM result
either way -- the synthetic run's only job is to catch that failure mode
before it contaminates the real-data conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _apply_nonlinearity(z: np.ndarray, nonlinearity: str) -> np.ndarray:
    if nonlinearity == "identity":
        return z
    if nonlinearity == "tanh":
        return np.tanh(z)
    if nonlinearity == "polynomial_3":
        return z + 0.1 * z ** 3
    raise ValueError(f"unknown nonlinearity: {nonlinearity!r}")


@dataclass
class SyntheticTransitionGenerator:
    """Parameterized nonlinear latent process -> (X_t, Y_{t+1}) transition pairs.

    Dynamics: `z_{t+1} = g(z_t) + J @ g(z_t) + noise`, where `g` is the
    configured elementwise `nonlinearity` and `J` is a fixed random coupling
    matrix scaled by `coupling_strength`. `coupling_strength=0` makes every
    node's dynamics independent (a harness sanity floor); higher values
    introduce genuine cross-node structure for e.g. SINDy's pairwise terms
    or DMD's off-diagonal operator entries to (in principle) recover.

    Parameters
    ----------
    num_nodes : int
        Number of scalar observables (default 8, matching A1's real
        mass/chromosome node set for direct shape comparability).
    coupling_strength : float
        Scale of the cross-node coupling matrix `J` (0 = independent nodes).
    nonlinearity : str
        `"identity"`, `"tanh"`, or `"polynomial_3"`.
    process_noise_std : float
        Additive noise on each step -- this is what caps the achievable R2
        below 1.0; the ceiling is measured, not set directly.
    seed : int
        Random seed for both the coupling matrix and generated trajectories.
    """

    num_nodes: int = 8
    coupling_strength: float = 0.3
    nonlinearity: str = "tanh"
    process_noise_std: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._J = (
            rng.normal(size=(self.num_nodes, self.num_nodes)) * self.coupling_strength / self.num_nodes
            if self.coupling_strength > 0
            else np.zeros((self.num_nodes, self.num_nodes))
        )

    def _drift(self, z: np.ndarray) -> np.ndarray:
        g = _apply_nonlinearity(z, self.nonlinearity)
        return g + g @ self._J.T

    def _simulate_trajectory(self, rng: np.random.Generator, steps: int) -> np.ndarray:
        z = rng.normal(scale=0.5, size=self.num_nodes)
        traj = np.empty((steps + 1, self.num_nodes))
        traj[0] = z
        for t in range(steps):
            drift = self._drift(traj[t])
            noise = rng.normal(scale=self.process_noise_std, size=self.num_nodes)
            traj[t + 1] = drift + noise
        return traj

    def _measure_achievable_r2(self, rng: np.random.Generator, n_samples: int) -> dict[str, float]:
        """Measure (not assert) the best-possible per-node R2: run the exact
        noiseless drift on real trajectory states, compare its residual
        variance against fresh independent noise draws, on a batch large
        enough that the estimate itself is stable.
        """
        calib_traj = self._simulate_trajectory(rng, n_samples)
        X = calib_traj[:-1]
        noiseless = np.stack([self._drift(X[i]) for i in range(X.shape[0])])
        noise = rng.normal(scale=self.process_noise_std, size=noiseless.shape)
        noisy = noiseless + noise

        ss_res = np.sum((noisy - noiseless) ** 2, axis=0)
        mu = noisy.mean(axis=0)
        ss_tot = np.sum((noisy - mu) ** 2, axis=0)
        valid = ss_tot > 0
        r2 = np.full(self.num_nodes, np.nan)
        r2[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
        return {f"x{i}": float(r2[i]) for i in range(self.num_nodes)}

    def generate(self, num_trajectories: int = 4, steps_per_trajectory: int = 200,
                 calibration_steps: int = 5000) -> dict[str, Any]:
        """Generate transition pairs matching `build_transition_pairs()`'s
        `(X, Y, traj_id)` shape, plus a measured `achievable_r2` per node.

        `calibration_steps` controls only the achievable-R2 measurement's
        precision -- it is independent of (and typically much larger than)
        `num_trajectories * steps_per_trajectory`, so the reported ceiling
        isn't itself noisy from the small returned dataset.
        """
        rng = np.random.default_rng(self.seed)
        xs, ys, tids = [], [], []
        for traj_id in range(num_trajectories):
            traj = self._simulate_trajectory(rng, steps_per_trajectory)
            xs.append(traj[:-1])
            ys.append(traj[1:])
            tids.append(np.full(steps_per_trajectory, traj_id, dtype=np.int64))

        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        traj_id_arr = np.concatenate(tids, axis=0)
        achievable_r2 = self._measure_achievable_r2(rng, calibration_steps)

        return {
            "X": X,
            "Y": Y,
            "traj_id": traj_id_arr,
            "achievable_r2": achievable_r2,
            "config": {
                "num_nodes": self.num_nodes,
                "coupling_strength": self.coupling_strength,
                "nonlinearity": self.nonlinearity,
                "process_noise_std": self.process_noise_std,
                "seed": self.seed,
                "num_trajectories": num_trajectories,
                "steps_per_trajectory": steps_per_trajectory,
            },
        }
