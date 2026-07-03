"""Tests for the A1 QGRNN-style surrogate and its classical-GNN baseline.

Forward-pass shape/sanity tests only, no full-training test here -- per
todo.md's testing plan, the real training run belongs in
docs/investigation_a1_qgrnn_surrogate/run_qgrnn_surrogate.py, not pytest.
"""

from __future__ import annotations

import pytest
import torch

from pbg_pennylane_adversarial.qgrnn_surrogate import (
    ClassicalGNNSurrogate,
    QGRNNSurrogate,
    fully_connected_edges,
)


def _triangle_edges():
    return [(0, 1), (0, 2), (1, 2)]


class TestFullyConnectedEdges:
    def test_edge_count_and_pairs(self):
        edges = fully_connected_edges(4)
        assert len(edges) == 6
        assert set(edges) == {
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
        }


class TestQGRNNSurrogate:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = QGRNNSurrogate(num_nodes=3, edges=_triangle_edges(), trotter_steps=2)
        x = torch.rand(4, 3)
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        model = QGRNNSurrogate(num_nodes=3, edges=_triangle_edges(), trotter_steps=2)
        x = torch.rand(2, 3)
        out = model(x)
        loss = out.pow(2).sum()
        loss.backward()
        assert torch.isfinite(out).all()
        assert model.zz_weights.grad is not None
        assert model.x_weights.grad is not None

    def test_l1_penalty_matches_zz_weights(self):
        model = QGRNNSurrogate(num_nodes=3, edges=_triangle_edges())
        assert torch.allclose(model.l1_penalty(), model.zz_weights.abs().sum())

    def test_coupling_matrix_symmetric_and_matches_edges(self):
        model = QGRNNSurrogate(num_nodes=3, edges=_triangle_edges())
        mat = model.coupling_matrix()
        assert mat.shape == (3, 3)
        assert torch.allclose(mat, mat.T)
        assert torch.all(mat.diagonal() == 0)
        assert mat[0, 1] == pytest.approx(model.zz_weights[0].abs().item())

    def test_invalid_edge_raises(self):
        with pytest.raises(ValueError):
            QGRNNSurrogate(num_nodes=3, edges=[(0, 0)])
        with pytest.raises(ValueError):
            QGRNNSurrogate(num_nodes=3, edges=[(0, 5)])


class TestClassicalGNNSurrogate:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = ClassicalGNNSurrogate(num_nodes=3, edges=_triangle_edges())
        x = torch.rand(5, 3)
        out = model(x)
        assert out.shape == (5, 3)

    def test_output_is_finite_and_differentiable(self):
        torch.manual_seed(0)
        model = ClassicalGNNSurrogate(num_nodes=3, edges=_triangle_edges())
        x = torch.rand(5, 3)
        out = model(x)
        loss = out.pow(2).sum()
        loss.backward()
        assert torch.isfinite(out).all()
        assert model.edge_weights.grad is not None
        assert model.node_weights.grad is not None

    def test_l1_penalty_matches_edge_weights(self):
        model = ClassicalGNNSurrogate(num_nodes=3, edges=_triangle_edges())
        assert torch.allclose(model.l1_penalty(), model.edge_weights.abs().sum())

    def test_coupling_matrix_symmetric_and_matches_edges(self):
        model = ClassicalGNNSurrogate(num_nodes=3, edges=_triangle_edges())
        mat = model.coupling_matrix()
        assert mat.shape == (3, 3)
        assert torch.allclose(mat, mat.T)
        assert torch.all(mat.diagonal() == 0)
        assert mat[0, 1] == pytest.approx(model.edge_weights[0].abs().item())

    def test_invalid_edge_raises(self):
        with pytest.raises(ValueError):
            ClassicalGNNSurrogate(num_nodes=3, edges=[(1, 1)])

    def test_trains_to_lower_loss_on_synthetic_linear_system(self):
        # Cheap (pure torch, no circuit) sanity check that the message-passing
        # architecture is actually trainable, not just shape-correct.
        torch.manual_seed(0)
        edges = fully_connected_edges(4)
        A = torch.randn(4, 4) * 0.2
        X = torch.rand(64, 4) * 2 - 1
        Y = torch.tanh(X @ A.T)

        model = ClassicalGNNSurrogate(num_nodes=4, edges=edges, num_steps=2, seed=1)
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()

        initial_loss = loss_fn(model(X), Y).item()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(model(X), Y)
            loss.backward()
            opt.step()
        final_loss = loss_fn(model(X), Y).item()

        assert final_loss < initial_loss * 0.5
