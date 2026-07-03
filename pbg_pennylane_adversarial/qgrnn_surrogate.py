"""QGRNN-style graph-structured surrogate for WCM transition dynamics (A1).

Two architectures over the same node/edge set, trained on (X_t, Y_{t+1})
transition pairs from ``wcm_loader.build_transition_pairs()``: a QGRNN-style
quantum circuit (one qubit per node, Trotterized Ising-style Hamiltonian --
one ZZ coupling term per graph edge, one transverse-field RX per node) and a
same-topology classical GNN (one scalar coupling per edge, one scalar field
per node, tanh nonlinearity in place of unitary evolution). The two share a
parameterization (O(edges) + O(nodes), not a dense per-layer weight matrix)
so a QGRNN-vs-MLP comparison alone can't be confounded with "graph structure
helped" vs. "the quantum circuit specifically helped" (see todo.md's
"Novelty -- deep dive").

Both expose an L1 edge-sparsity penalty (``l1_penalty()``) so training can
turn the dense starting graph into a sparsified, inferred topology rather
than a dense heatmap of near-equal weights, and a ``coupling_matrix()`` for
plotting/interpretability checks against known bacterial-physiology
relationships.

Callers should z-score normalize input features before calling either model:
both encode raw input values directly as circuit rotation angles / initial
node states, and unnormalized WCM features (e.g. cell_mass ~ 500) would make
angle encoding badly conditioned.
"""

from __future__ import annotations

import pennylane as qml
import torch


def fully_connected_edges(num_nodes: int) -> list[tuple[int, int]]:
    """The default graph for A1's first pass (todo.md: start fully-connected,
    let L1 sparsity suppress unimportant edges, rather than presupposing a
    topology by hand-curating a chain/star)."""
    return [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]


def _validate_edges(num_nodes: int, edges: list[tuple[int, int]]) -> None:
    for i, j in edges:
        if i == j or not (0 <= i < num_nodes and 0 <= j < num_nodes):
            raise ValueError(f"invalid edge {(i, j)} for num_nodes={num_nodes}")


def _coupling_matrix(num_nodes: int, edges: list[tuple[int, int]],
                      weights: torch.Tensor) -> torch.Tensor:
    mat = torch.zeros(num_nodes, num_nodes)
    for (i, j), w in zip(edges, weights.detach().abs()):
        mat[i, j] = w
        mat[j, i] = w
    return mat


class QGRNNSurrogate(torch.nn.Module):
    """Recurrent graph-structured quantum circuit (QGRNN-style, per Verdon et
    al. 2019 / PennyLane's ``tutorial_qgrnn``), trained via MSE regression
    against real ``Y_{t+1}`` rather than the original tutorial's fidelity/
    self-consistency cost -- the surrogate use case is regression, not
    Hamiltonian-learning-by-fidelity.
    """

    def __init__(self, num_nodes: int, edges: list[tuple[int, int]],
                 trotter_steps: int = 3, dt: float = 0.5, seed: int = 0):
        super().__init__()
        _validate_edges(num_nodes, edges)
        self.num_nodes = num_nodes
        self.edges = list(edges)
        self.trotter_steps = trotter_steps
        self.dt = dt

        gen = torch.Generator().manual_seed(seed)
        self.zz_weights = torch.nn.Parameter(
            torch.randn(len(self.edges), generator=gen) * 0.1
        )
        self.x_weights = torch.nn.Parameter(
            torch.randn(num_nodes, generator=gen) * 0.1
        )

        dev = qml.device("lightning.qubit", wires=num_nodes)
        edges_, n_, steps_, dt_ = self.edges, num_nodes, trotter_steps, dt

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, zz_weights, x_weights):
            for i in range(n_):
                qml.RY(inputs[i], wires=i)
            for _ in range(steps_):
                for edge_idx, (i, j) in enumerate(edges_):
                    qml.IsingZZ(2 * dt_ * zz_weights[edge_idx], wires=[i, j])
                for i in range(n_):
                    qml.RX(2 * dt_ * x_weights[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-sample circuit evaluation: TorchLayer-style batching doesn't apply
        # here since the circuit takes two separate weight tensors alongside the
        # per-sample input (validated in the phase-2 circuit spike).
        outs = [
            torch.stack(self._circuit(x[b], self.zz_weights, self.x_weights))
            for b in range(x.shape[0])
        ]
        return torch.stack(outs)

    def l1_penalty(self) -> torch.Tensor:
        return self.zz_weights.abs().sum()

    def coupling_matrix(self) -> torch.Tensor:
        return _coupling_matrix(self.num_nodes, self.edges, self.zz_weights)


class ClassicalGNNSurrogate(torch.nn.Module):
    """Same-topology classical baseline -- isolates whether a QGRNN's edge (if
    any) over an unstructured MLP comes from having graph structure at all
    (this model) vs. the quantum circuit's specific parameterization of that
    structure (QGRNNSurrogate). Message-passing steps mirror QGRNN's Trotter
    steps; tanh stands in for unitary evolution as the nonlinearity.
    """

    def __init__(self, num_nodes: int, edges: list[tuple[int, int]],
                 num_steps: int = 3, dt: float = 0.5, seed: int = 0):
        super().__init__()
        _validate_edges(num_nodes, edges)
        self.num_nodes = num_nodes
        self.edges = list(edges)
        self.num_steps = num_steps
        self.dt = dt

        gen = torch.Generator().manual_seed(seed)
        self.edge_weights = torch.nn.Parameter(
            torch.randn(len(self.edges), generator=gen) * 0.1
        )
        self.node_weights = torch.nn.Parameter(
            torch.randn(num_nodes, generator=gen) * 0.1
        )
        self.register_buffer(
            "_edge_i", torch.tensor([i for i, _ in self.edges], dtype=torch.long)
        )
        self.register_buffer(
            "_edge_j", torch.tensor([j for _, j in self.edges], dtype=torch.long)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.num_steps):
            h_i = h[:, self._edge_i]
            h_j = h[:, self._edge_j]
            msg = torch.zeros(h.shape[0], self.num_nodes, dtype=h.dtype)
            msg = msg.index_add(1, self._edge_i, self.edge_weights * h_j)
            msg = msg.index_add(1, self._edge_j, self.edge_weights * h_i)
            h = h + self.dt * torch.tanh(msg + self.node_weights * h)
        return h

    def l1_penalty(self) -> torch.Tensor:
        return self.edge_weights.abs().sum()

    def coupling_matrix(self) -> torch.Tensor:
        return _coupling_matrix(self.num_nodes, self.edges, self.edge_weights)
