#!/usr/bin/env python3
"""Generate demo report for pbg-pennylane-adversarial.

Runs three configurations of the adversarial attack pipeline and produces
a self-contained HTML report with Plotly charts and bigraph-viz2 diagrams.
"""

import os
import sys
import json
import time
import webbrowser

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.html")


def build_core():
    core = allocate_core()
    core.register_link("ram-emitter", RAMEmitter)
    return core


def gather_emitter_results(sim):
    """Gather results from the RAMEmitter after a simulation run."""
    from process_bigraph.emitter import gather_emitter_results as _gather_results
    return _gather_results(sim)


CONFIGS = [
    {
        "id": "baseline",
        "title": "Baseline Pipeline",
        "subtitle": "4 training epochs, epsilon=0.1, 2 adversarial epochs",
        "description": (
            "Standard adversarial attack pipeline with a 8-qubit, 32-layer "
            "data-reuploading QML classifier. The model is trained on the "
            "PlusMinus dataset, then attacked with PGD (epsilon=0.1, 10 "
            "iterations), and finally retrained with adversarial examples."
        ),
        "config": {
            "num_qubits": 8,
            "num_layers": 8,
            "training_epochs": 2,
            "adversarial_epochs": 1,
            "pgd_iter": 5,
            "n_train": 50,
            "n_test": 20,
            "batch_size": 10,
            "seed": 1337,
        },
        "accent": "#636efa",
        "n_snapshots": 15,
        "total_time": 15.0,
        "dataset": {
            "name": "PlusMinus",
            "source": "PennyLane built-in (other/plus-minus)",
            "n_train": 50,
            "n_test": 20,
            "input_dim": 784,
            "output_dim": 2,
        },
    },
    {
        "id": "strong_attack",
        "title": "Strong Attack",
        "subtitle": "Wider epsilon=0.2, more PGD iterations (20)",
        "description": (
            "Same architecture as baseline but with a stronger PGD attack "
            "(epsilon=0.2, 20 iterations). Tests the model's robustness "
            "against more aggressive perturbations."
        ),
        "config": {
            "num_qubits": 8,
            "num_layers": 8,
            "training_epochs": 2,
            "adversarial_epochs": 2,
            "pgd_iter": 10,
            "epsilon": 0.2,
            "n_train": 50,
            "n_test": 20,
            "batch_size": 10,
            "seed": 1337,
        },
        "accent": "#ef553b",
        "n_snapshots": 20,
        "total_time": 20.0,
        "dataset": {
            "name": "PlusMinus",
            "source": "PennyLane built-in (other/plus-minus)",
            "n_train": 50,
            "n_test": 20,
            "input_dim": 784,
            "output_dim": 2,
        },
    },
    {
        "id": "lightweight",
        "title": "Lightweight",
        "subtitle": "Small circuit (4 qubits, 4 layers) for fast iteration",
        "description": (
            "Reduced model with 4 qubits and 4 layers. Uses fewer training "
            "samples and fewer epochs. Designed for quick experimentation "
            "and CI runs."
        ),
        "config": {
            "num_qubits": 4,
            "num_layers": 4,
            "training_epochs": 1,
            "adversarial_epochs": 1,
            "pgd_iter": 3,
            "n_train": 30,
            "n_test": 10,
            "batch_size": 10,
            "seed": 42,
        },
        "accent": "#00cc96",
        "n_snapshots": 10,
        "total_time": 10.0,
        "dataset": {
            "name": "PlusMinus",
            "source": "PennyLane built-in (other/plus-minus)",
            "n_train": 30,
            "n_test": 10,
            "input_dim": 784,
            "output_dim": 2,
        },
    },
]


def run_config(cfg):
    """Run a single config and return metrics time series."""
    from pbg_pennylane_adversarial import PennyLaneAdversarialProcess

    print(f"  Running '{cfg['id']}'...", end=" ", flush=True)
    t0 = time.perf_counter()

    core = build_core()
    proc = PennyLaneAdversarialProcess(config=cfg["config"], core=core)
    state = proc.initial_state()

    records = []
    max_steps = cfg["total_time"]

    for step in range(int(max_steps)):
        result = proc.update(state, interval=1.0)
        state.update(result)
        records.append({
            "step": step,
            "phase": state["phase"],
            "epoch": state["epoch"],
            "loss": state["loss"],
            "accuracy": state["accuracy"],
            "benign_accuracy": state["benign_accuracy"],
            "adversarial_accuracy": state["adversarial_accuracy"],
            "robust_accuracy": state["robust_accuracy"],
            "adversarial_accuracy_drop": state["adversarial_accuracy_drop"],
            "n_queries": state["n_queries"],
        })
        if state["phase"] == "done":
            break

    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s, {len(records)} steps)")
    return records, elapsed


def build_metrics_card(title, value, subtitle, color):
    """Return HTML for a single metrics card."""
    return f"""
    <div style="background:{color}15;border-radius:12px;padding:16px;
                border-left:4px solid {color};flex:1;min-width:140px;">
      <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                  letter-spacing:0.05em;margin-bottom:4px;">{title}</div>
      <div style="font-size:1.5rem;font-weight:700;color:#1e293b;">{value}</div>
      <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">{subtitle}</div>
    </div>"""


def build_dataset_badge(ds):
    """Return HTML for a dataset info badge."""
    return f"""
    <div style="background:#eef2ff;border-radius:8px;padding:12px;
                border:1px solid #c7d2fe;margin-bottom:16px;">
      <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;
                  letter-spacing:0.05em;margin-bottom:4px;">Dataset</div>
      <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;">
        <span style="font-weight:700;color:#4338ca;font-size:1rem;">{ds['name']}</span>
        <span style="font-size:0.8rem;color:#64748b;">
          {ds.get('source', 'custom')} &mdash;
          {ds['n_train']} train / {ds['n_test']} test samples,
          {ds['input_dim']}-d features,
          {ds['output_dim']} classes
        </span>
      </div>
    </div>"""


def build_section(records, cfg):
    """Build an HTML section for one config."""
    accent = cfg["accent"]

    if not records:
        return f"<h3>{cfg['title']}</h3><p>No data (config error).</p>"

    last = records[-1]

    final_phase = last["phase"]
    benign_acc = last.get("benign_accuracy", 0.0)
    adv_acc = last.get("adversarial_accuracy", 0.0)
    robust_acc = last.get("robust_accuracy", 0.0)
    drop = last.get("adversarial_accuracy_drop", 0.0)
    n_q = last.get("n_queries", 0)

    cards_html = "".join([
        build_metrics_card("Benign Accuracy", f"{benign_acc:.1%}",
                           "Clean test set", "#22c55e"),
        build_metrics_card("Adversarial Accuracy", f"{adv_acc:.1%}",
                           "Under PGD attack", accent),
        build_metrics_card("Robust Accuracy", f"{robust_acc:.1%}",
                           "After adversarial retraining", "#636efa"),
        build_metrics_card("Accuracy Drop", f"{drop:.1%}",
                           "Benign → adversarial", "#ef4444"),
        build_metrics_card("Phase", final_phase,
                           "Pipeline stage", "#8b5cf6"),
        build_metrics_card("Circuit Evals", f"{n_q:,}",
                           "Total QPU calls", "#f59e0b"),
    ])

    dataset_html = build_dataset_badge(cfg.get("dataset", {
        "name": "Custom",
        "source": "Supplied through input ports",
        "n_train": "?",
        "n_test": "?",
        "input_dim": "auto",
        "output_dim": "auto",
    }))

    # Accuracy chart (Plotly)
    steps = [r["step"] for r in records]
    benign_series = [r.get("benign_accuracy", 0.0) for r in records]
    adv_series = [r.get("adversarial_accuracy", 0.0) for r in records]
    robust_series = [r.get("robust_accuracy", 0.0) for r in records]
    loss_series = [r.get("loss", 0.0) for r in records]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Accuracy vs. Step", "Loss vs. Step"),
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Scatter(
        x=steps, y=benign_series, mode="lines+markers",
        name="Benign", line=dict(color="#22c55e", width=2),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=adv_series, mode="lines+markers",
        name="Adversarial", line=dict(color=accent, width=2, dash="dot"),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=robust_series, mode="lines+markers",
        name="Robust", line=dict(color="#636efa", width=2, dash="dash"),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=steps, y=loss_series, mode="lines+markers",
        name="Loss", line=dict(color="#f59e0b", width=2),
        marker=dict(size=6),
    ), row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, range=[0, 1.05])
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)

    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

    # Phase timeline
    phases_seen = []
    for r in records:
        p = r.get("phase", "?")
        if not phases_seen or phases_seen[-1] != p:
            phases_seen.append(p)

    phases_html = " → ".join(
        f'<span style="background:{accent}22;color:{accent};padding:2px 8px;'
        f'border-radius:4px;font-size:0.8rem;">{p}</span>'
        for p in phases_seen
    )

    return f"""
    <section id="{cfg['id']}" style="margin-bottom:48px;scroll-margin-top:70px;">
      <h2 style="font-size:1.3rem;font-weight:600;color:#1e293b;
                 border-bottom:3px solid {accent};padding-bottom:8px;
                 margin-bottom:16px;">
        {cfg['title']}
        <span style="font-size:0.85rem;font-weight:400;color:#64748b;
                     margin-left:12px;">{cfg['subtitle']}</span>
      </h2>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;margin-bottom:16px;">
        {cfg['description']}
      </p>
      {dataset_html}
      <div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px;">
        {cards_html}
      </div>
      <div style="margin-bottom:12px;">
        <strong style="color:#475569;font-size:0.8rem;">Pipeline phases:</strong>
        <div style="margin-top:6px;">{phases_html}</div>
      </div>
      {chart_html}
      <details style="margin-top:12px;">
        <summary style="cursor:pointer;color:{accent};font-weight:500;
                       font-size:0.85rem;">
          Show raw metrics table
        </summary>
        <table style="width:100%;border-collapse:collapse;font-size:0.75rem;
                      margin-top:8px;">
          <thead>
            <tr style="background:#f1f5f9;">
              <th style="padding:6px 8px;text-align:left;">Step</th>
              <th style="padding:6px 8px;text-align:left;">Phase</th>
              <th style="padding:6px 8px;text-align:right;">Loss</th>
              <th style="padding:6px 8px;text-align:right;">Accuracy</th>
              <th style="padding:6px 8px;text-align:right;">Benign</th>
              <th style="padding:6px 8px;text-align:right;">Adv</th>
              <th style="padding:6px 8px;text-align:right;">Robust</th>
              <th style="padding:6px 8px;text-align:right;">Q Evals</th>
            </tr>
          </thead>
          <tbody>
            {''.join(
              f'<tr style="border-bottom:1px solid #e2e8f0;">'
              f'<td style="padding:4px 8px;">{r["step"]}</td>'
              f'<td style="padding:4px 8px;">{r["phase"]}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["loss"]:.4f}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["accuracy"]:.4f}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["benign_accuracy"]:.4f}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["adversarial_accuracy"]:.4f}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["robust_accuracy"]:.4f}</td>'
              f'<td style="padding:4px 8px;text-align:right;">{r["n_queries"]}</td>'
              f'</tr>'
              for r in records
            )}
          </tbody>
        </table>
      </details>
    </section>
    """


def generate_report():
    """Main report generation entry point."""
    print("=" * 60)
    print("pbg-pennylane-adversarial Demo Report")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_times = {}
    all_records = {}

    for cfg in CONFIGS:
        try:
            records, elapsed = run_config(cfg)
            all_records[cfg["id"]] = records
            all_times[cfg["id"]] = elapsed
        except Exception as e:
            print(f"  FAILED: {e}")
            all_records[cfg["id"]] = []
            all_times[cfg["id"]] = 0.0

    print("\nGenerating report...")

    # Build sections
    sections_html = "".join(
        build_section(all_records.get(cfg["id"], []), cfg)
        for cfg in CONFIGS
    )

    # Summary cards
    total_time = sum(all_times.values())
    total_evals = sum(
        r[-1]["n_queries"] if r else 0
        for r in all_records.values()
    )
    total_steps = sum(len(r) for r in all_records.values())

    # Architecture summary
    arch_html = f"""
    <section id="architecture" style="margin-bottom:48px;scroll-margin-top:70px;">
      <h2 style="font-size:1.3rem;font-weight:600;color:#1e293b;
                 border-bottom:3px solid #8b5cf6;padding-bottom:8px;">
        Architecture
      </h2>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;">
        The <code>PennyLaneAdversarialProcess</code> wraps a QML classifier built with
        <strong>PennyLane's</strong> <code>StronglyEntanglingLayers</code> template
        and <code>TorchLayer</code> integration. The pipeline progresses through:
      </p>
      <ul style="color:#475569;font-size:0.9rem;line-height:1.8;">
        <li><strong>Training</strong> — supervised cross-entropy optimization on the
          supplied dataset (Adam optimizer).</li>
        <li><strong>Benign evaluation</strong> — accuracy on clean test samples.</li>
        <li><strong>PGD Attack</strong> — projected gradient descent perturbation
          (epsilon-bound L_inf, sign-gradient ascent).</li>
        <li><strong>Adversarial evaluation</strong> — accuracy under PGD attack.</li>
        <li><strong>Adversarial retraining</strong> — training with augmented dataset
          containing original + perturbed samples.</li>
        <li><strong>Robust evaluation</strong> — accuracy on perturbed data after
          retraining.</li>
      </ul>

      <h3 style="font-size:1rem;font-weight:600;color:#1e293b;margin-top:20px;
                 margin-bottom:8px;">Data Interface</h3>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;">
        Data arrives through four input ports:
      </p>
      <table style="width:100%;border-collapse:collapse;font-size:0.85rem;
                    margin:8px 0;">
        <thead>
          <tr style="background:#f1f5f9;">
            <th style="padding:6px 10px;text-align:left;">Port</th>
            <th style="padding:6px 10px;text-align:left;">Type</th>
            <th style="padding:6px 10px;text-align:left;">Shape</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:6px 10px;"><code>train_images</code></td>
            <td style="padding:6px 10px;"><code>array[float64]</code></td>
            <td style="padding:6px 10px;">(n_train, input_dim)</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:6px 10px;"><code>train_labels</code></td>
            <td style="padding:6px 10px;"><code>array[int64]</code></td>
            <td style="padding:6px 10px;">(n_train,)</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:6px 10px;"><code>test_images</code></td>
            <td style="padding:6px 10px;"><code>array[float64]</code></td>
            <td style="padding:6px 10px;">(n_test, input_dim)</td>
          </tr>
          <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:6px 10px;"><code>test_labels</code></td>
            <td style="padding:6px 10px;"><code>array[int64]</code></td>
            <td style="padding:6px 10px;">(n_test,)</td>
          </tr>
        </tbody>
      </table>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;">
        When no data is wired (<code>train_images</code> is empty or missing),
        the process automatically falls back to PennyLane's built-in
        <strong>PlusMinus</strong> dataset for backward compatibility.
      </p>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;margin-top:8px;">
        <code>input_dim</code> and <code>output_dim</code> are auto-detected from
        the data shape (number of columns, number of unique labels respectively)
        but can be overridden via config. The circuit architecture adapts
        dynamically: <code>num_reup</code> (data re-uploading repeats) is computed
        to satisfy the <code>StronglyEntanglingLayers</code> weight tensor
        dimension constraint.
      </p>
      <p style="color:#475569;font-size:0.9rem;line-height:1.5;margin-top:8px;">
        Each <code>update()</code> advances one epoch or evaluation phase.
        Data is cached internally on first call, so forward/backward passes
        reuse tensors without re-reading stores each step.
      </p>
    </section>
    """

    # Generate the bigraph diagram using bigraph-viz2
    bigraph_html = ""
    try:
        from bigraph_viz2 import emit_html

        doc = {
            "adversarial": {
                "_type": "process",
                "address": "local:PennyLaneAdversarialProcess",
                "config": {"num_qubits": 8},
                "inputs": {},
                "outputs": {
                    "phase": ["stores", "phase"],
                    "accuracy": ["stores", "accuracy"],
                    "loss": ["stores", "loss"],
                    "benign_accuracy": ["stores", "benign_accuracy"],
                    "adversarial_accuracy": ["stores", "adversarial_accuracy"],
                    "robust_accuracy": ["stores", "robust_accuracy"],
                    "n_queries": ["stores", "n_queries"],
                },
            },
            "stores": {
                "phase": "init",
                "accuracy": 0.0,
                "loss": 0.0,
                "benign_accuracy": 0.0,
                "adversarial_accuracy": 0.0,
                "robust_accuracy": 0.0,
                "n_queries": 0,
            },
            "emitter": {
                "_type": "step",
                "address": "local:RAMEmitter",
                "config": {"emit": {
                    "phase": "string",
                    "accuracy": "float",
                    "loss": "float",
                    "benign_accuracy": "float",
                    "adversarial_accuracy": "float",
                    "robust_accuracy": "float",
                    "n_queries": "integer",
                }},
                "inputs": {
                    "phase": ["stores", "phase"],
                    "accuracy": ["stores", "accuracy"],
                    "loss": ["stores", "loss"],
                    "benign_accuracy": ["stores", "benign_accuracy"],
                    "adversarial_accuracy": ["stores", "adversarial_accuracy"],
                    "robust_accuracy": ["stores", "robust_accuracy"],
                    "n_queries": ["stores", "n_queries"],
                    "time": ["global_time"],
                },
            },
        }
        bigraph_html = emit_html(doc, height="400px", inspector=True, dedupe=False)
    except ImportError:
        bigraph_html = '<p style="color:#94a3b8;">bigraph-viz2 not available</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-pennylane-adversarial — Demo Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:#f8fafc;color:#1e293b;line-height:1.5; }}
nav {{ position:sticky;top:0;z-index:100;background:#ffffffdd;backdrop-filter:blur(8px);
       border-bottom:1px solid #e2e8f0;padding:12px 24px;
       display:flex;align-items:center;gap:24px;flex-wrap:wrap; }}
nav a {{ color:#636efa;text-decoration:none;font-size:0.85rem;font-weight:500; }}
nav a:hover {{ text-decoration:underline; }}
.container {{ max-width:1100px;margin:0 auto;padding:24px; }}
h1 {{ font-size:1.6rem;font-weight:700;margin-bottom:4px; }}
.subtitle {{ color:#64748b;font-size:0.9rem;margin-bottom:24px; }}
.summary-grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:12px;margin-bottom:32px; }}
code {{ background:#f1f5f9;padding:2px 6px;border-radius:4px;
        font-size:0.85em; }}
table {{ width:100%;border-collapse:collapse; }}
th, td {{ padding:8px;text-align:left;border-bottom:1px solid #e2e8f0; }}
th {{ background:#f1f5f9;font-weight:600;font-size:0.85rem; }}
tr:hover {{ background:#f8fafc; }}
</style>
</head>
<body>
<nav>
  <strong style="color:#1e293b;">pbg-pennylane-adversarial</strong>
  {''.join(f'<a href="#{cfg["id"]}">{cfg["title"]}</a>' for cfg in CONFIGS)}
  <a href="#architecture">Architecture</a>
</nav>
<div class="container">
  <h1>Demo Report</h1>
  <p class="subtitle">
    PennyLane adversarial attacks on QML classifiers — PGD attack,
    adversarial retraining, and robustness evaluation.
  </p>

  <div class="summary-grid">
    {build_metrics_card("Configs Run", str(len(CONFIGS)), "Total configurations", "#636efa")}
    {build_metrics_card("Total Steps", str(total_steps), "Pipeline update() calls", "#8b5cf6")}
    {build_metrics_card("Circuit Evals", f"{total_evals:,}", "Total QPU calls", "#f59e0b")}
    {build_metrics_card("Wall Time", f"{total_time:.1f}s", "Total runtime", "#22c55e")}
  </div>

  {arch_html}

  <section id="bigraph" style="margin-bottom:48px;scroll-margin-top:70px;">
    <h2 style="font-size:1.3rem;font-weight:600;color:#1e293b;
               border-bottom:3px solid #8b5cf6;padding-bottom:8px;margin-bottom:16px;">
      Composite Architecture
    </h2>
    {bigraph_html}
  </section>

  {sections_html}

  <footer style="border-top:1px solid #e2e8f0;padding:24px 0;margin-top:32px;
                 text-align:center;color:#94a3b8;font-size:0.8rem;">
    Generated by pbg-pennylane-adversarial demo report generator.
  </footer>
</div>
</body>
</html>"""

    with open(REPORT_PATH, "w") as f:
        f.write(html)

    print(f"\nReport written to: {REPORT_PATH}")
    print(f"Total demo wall time: {total_time:.1f}s")

    return REPORT_PATH


if __name__ == "__main__":
    report_path = generate_report()
    webbrowser.open("file://" + os.path.abspath(report_path))
