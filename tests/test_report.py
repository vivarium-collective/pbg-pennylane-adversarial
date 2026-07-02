"""Tests for HTML report generation, including the baseline comparison section."""

from __future__ import annotations

from pbg_pennylane_adversarial.report import generate_run_report

RECORDS = [
    {
        "step": 0, "phase": "training", "epoch": 1, "loss": 0.5, "accuracy": 0.6,
        "benign_accuracy": 0.6, "adversarial_accuracy": 0.0, "robust_accuracy": 0.0,
        "adversarial_accuracy_drop": 0.0, "n_queries": 10,
    },
    {
        "step": 1, "phase": "done", "epoch": 0, "loss": 0.0, "accuracy": 0.85,
        "benign_accuracy": 0.85, "adversarial_accuracy": 0.7, "robust_accuracy": 0.8,
        "adversarial_accuracy_drop": 0.15, "n_queries": 42,
    },
]

DATASET_INFO = {"n_train": 40, "n_test": 10, "input_dim": 3, "output_dim": 2, "label_map": {}}
CONFIG = {"num_qubits": 4}


class TestGenerateRunReport:
    def test_no_baselines_omits_section(self):
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0)
        assert "Baseline Comparison" not in html

    def test_baselines_render_section(self):
        baselines = {
            "logistic_regression": {"benign_accuracy": 0.8, "adversarial_accuracy": 0.6},
            "random_forest": {"benign_accuracy": 0.82, "adversarial_accuracy": 0.65},
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "Baseline Comparison" in html
        assert "Logistic Regression" in html
        assert "Random Forest" in html

    def test_competitive_note_within_5pp(self):
        baselines = {
            "logistic_regression": {"benign_accuracy": 0.83, "adversarial_accuracy": 0.6},
            "random_forest": {"benign_accuracy": 0.80, "adversarial_accuracy": 0.6},
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "competitive" in html

    def test_trails_note_beyond_5pp(self):
        baselines = {
            "logistic_regression": {"benign_accuracy": 0.99, "adversarial_accuracy": 0.9},
            "random_forest": {"benign_accuracy": 0.95, "adversarial_accuracy": 0.9},
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "trails" in html

    def test_outperforms_note(self):
        baselines = {
            "logistic_regression": {"benign_accuracy": 0.5, "adversarial_accuracy": 0.4},
            "random_forest": {"benign_accuracy": 0.55, "adversarial_accuracy": 0.4},
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "outperforms" in html

    def test_empty_records(self):
        html = generate_run_report([], CONFIG, DATASET_INFO, wall_time=1.0)
        assert "No data" in html

    def test_transfer_accuracy_renders_when_present(self):
        baselines = {
            "logistic_regression": {
                "benign_accuracy": 0.8, "adversarial_accuracy": 0.6,
                "transfer_adversarial_accuracy": 0.75,
            },
            "random_forest": {
                "benign_accuracy": 0.82, "adversarial_accuracy": 0.65,
                "transfer_adversarial_accuracy": 0.78,
            },
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "Transfer attack" in html
        assert "75.0%" in html

    def test_no_transfer_accuracy_omits_transfer_section(self):
        baselines = {
            "logistic_regression": {"benign_accuracy": 0.8, "adversarial_accuracy": 0.6},
            "random_forest": {"benign_accuracy": 0.82, "adversarial_accuracy": 0.65},
        }
        html = generate_run_report(RECORDS, CONFIG, DATASET_INFO, wall_time=1.0,
                                    baselines=baselines)
        assert "Transfer attack" not in html
