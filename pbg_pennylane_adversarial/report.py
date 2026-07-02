"""HTML report generation for a single adversarial pipeline run."""

from __future__ import annotations


def _build_metrics_card(title, value, subtitle, color):
    return f"""
    <div style="background:{color}15;border-radius:12px;padding:16px;
                border-left:4px solid {color};flex:1;min-width:140px;">
      <div style="font-size:0.75rem;color:#64748b;text-transform:uppercase;
                  letter-spacing:0.05em;margin-bottom:4px;">{title}</div>
      <div style="font-size:1.5rem;font-weight:700;color:#1e293b;">{value}</div>
      <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">{subtitle}</div>
    </div>"""


def _build_dataset_badge(info):
    return f"""
    <div style="background:#eef2ff;border-radius:8px;padding:12px;
                border:1px solid #c7d2fe;margin-bottom:16px;">
      <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;
                  letter-spacing:0.05em;margin-bottom:4px;">Dataset</div>
      <div style="display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;">
        <span style="font-weight:700;color:#4338ca;font-size:1rem;">{info.get('source', 'Formatted artifact')}</span>
        <span style="font-size:0.8rem;color:#64748b;">
          {info['n_train']} train / {info['n_test']} test samples,
          {info['input_dim']}-d features,
          {info['output_dim']} classes
        </span>
      </div>
    </div>"""


def _build_config_badge(config):
    rows = "".join(
        f'<tr><td style="padding:2px 8px;color:#475569;">{k}</td>'
        f'<td style="padding:2px 8px;text-align:right;font-family:monospace;">{v}</td></tr>'
        for k, v in sorted(config.items())
    )
    return f"""
    <div style="background:#f8fafc;border-radius:8px;padding:12px;
                border:1px solid #e2e8f0;margin-bottom:16px;">
      <div style="font-size:0.75rem;color:#475569;text-transform:uppercase;
                  letter-spacing:0.05em;margin-bottom:4px;">Config</div>
      <table style="width:100%;font-size:0.8rem;border-collapse:collapse;">
        {rows}
      </table>
    </div>"""


def _build_accuracy_chart(records):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    steps = [r["step"] for r in records]
    benign = [r.get("benign_accuracy", 0.0) for r in records]
    adv = [r.get("adversarial_accuracy", 0.0) for r in records]
    robust = [r.get("robust_accuracy", 0.0) for r in records]
    loss = [r.get("loss", 0.0) for r in records]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Accuracy vs. Step", "Loss vs. Step"),
        vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(
        x=steps, y=benign, mode="lines+markers",
        name="Benign", line=dict(color="#22c55e", width=2),
        marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=adv, mode="lines+markers",
        name="Adversarial", line=dict(color="#ef553b", width=2, dash="dot"),
        marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=robust, mode="lines+markers",
        name="Robust", line=dict(color="#636efa", width=2, dash="dash"),
        marker=dict(size=6),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=steps, y=loss, mode="lines+markers",
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
    return fig.to_html(full_html=False, include_plotlyjs=False)


_BASELINE_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def _build_baseline_chart(qml_benign, qml_adversarial, baselines):
    import plotly.graph_objects as go

    labels = ["QML"] + [_BASELINE_LABELS.get(k, k) for k in baselines]
    benign = [qml_benign] + [v["benign_accuracy"] for v in baselines.values()]
    adversarial = [qml_adversarial] + [v["adversarial_accuracy"] for v in baselines.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Benign", x=labels, y=benign, marker_color="#22c55e"))
    fig.add_trace(go.Bar(name="Adversarial", x=labels, y=adversarial, marker_color="#ef553b"))
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        legend=dict(orientation="h", y=1.12),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _competitiveness_note(qml_benign, baselines):
    best_key, best = max(baselines.items(), key=lambda kv: kv[1]["benign_accuracy"])
    best_label = _BASELINE_LABELS.get(best_key, best_key)
    gap_pp = (qml_benign - best["benign_accuracy"]) * 100
    if abs(gap_pp) <= 5:
        return (f"QML is competitive with the best classical baseline "
                f"({best_label}, {best['benign_accuracy']:.1%}) — within "
                f"{abs(gap_pp):.1f} pp benign accuracy.")
    elif gap_pp > 0:
        return (f"QML outperforms the best classical baseline "
                f"({best_label}, {best['benign_accuracy']:.1%}) by "
                f"{gap_pp:.1f} pp benign accuracy.")
    else:
        return (f"QML trails the best classical baseline "
                f"({best_label}, {best['benign_accuracy']:.1%}) by "
                f"{abs(gap_pp):.1f} pp benign accuracy.")


def _build_baseline_section(qml_benign, qml_adversarial, baselines):
    if not baselines:
        return ""

    cards = "".join(
        _build_metrics_card(
            _BASELINE_LABELS.get(name, name),
            f"{metrics['benign_accuracy']:.1%} → {metrics['adversarial_accuracy']:.1%}",
            "benign → adversarial accuracy (own attack)",
            "#0ea5e9",
        )
        for name, metrics in baselines.items()
    )
    has_transfer = all("transfer_adversarial_accuracy" in m for m in baselines.values())
    transfer_cards = "".join(
        _build_metrics_card(
            _BASELINE_LABELS.get(name, name),
            f"{metrics['transfer_adversarial_accuracy']:.1%}",
            "accuracy under QML-crafted perturbation",
            "#a855f7",
        )
        for name, metrics in baselines.items()
    ) if has_transfer else ""
    chart_html = _build_baseline_chart(qml_benign, qml_adversarial, baselines)
    note = _competitiveness_note(qml_benign, baselines)

    transfer_section = f"""
  <p style="font-size:0.8rem;color:#475569;margin-bottom:8px;">
    Transfer attack — same perturbation the QML PGD attack produced, replayed against each baseline:
  </p>
  <div class="metrics-grid" style="margin-bottom:24px;">{transfer_cards}</div>""" if has_transfer else ""

    return f"""
  <h2 style="font-size:1.1rem;margin:8px 0 8px;">Baseline Comparison</h2>
  <p style="font-size:0.85rem;color:#475569;margin-bottom:12px;">{note}</p>
  <div class="metrics-grid" style="margin-bottom:12px;">{cards}</div>
  <div style="margin-bottom:24px;">{chart_html}</div>{transfer_section}"""


def generate_run_report(records, config, dataset_info, wall_time, accent="#636efa",
                         baselines=None):
    """Generate a self-contained HTML report for a single pipeline run.

    Parameters
    ----------
    records : list[dict]
        Per-step records with keys ``step``, ``phase``, ``epoch``, ``loss``,
        ``accuracy``, ``benign_accuracy``, ``adversarial_accuracy``,
        ``robust_accuracy``, ``adversarial_accuracy_drop``, ``n_queries``.
    config : dict
        Process configuration used for the run.
    dataset_info : dict
        Metadata with keys ``n_train``, ``n_test``, ``input_dim``, ``output_dim``,
        and optionally ``source``, ``label_map``.
    wall_time : float
        Wall-clock seconds for the run.
    accent : str
        CSS accent color for the report theme.
    baselines : dict, optional
        ``{"logistic_regression": {benign_accuracy, adversarial_accuracy},
        "random_forest": {...}}``. Renders a Baseline Comparison section
        when provided.

    Returns
    -------
    str
        Self-contained HTML string.
    """
    if not records:
        return "<html><body><p>No data — pipeline may have failed.</p></body></html>"

    last = records[-1]
    benign_acc = last.get("benign_accuracy", 0.0)
    adv_acc = last.get("adversarial_accuracy", 0.0)
    robust_acc = last.get("robust_accuracy", 0.0)
    drop = last.get("adversarial_accuracy_drop", 0.0)
    n_q = last.get("n_queries", 0)

    cards = "".join([
        _build_metrics_card("Benign Accuracy", f"{benign_acc:.1%}",
                            "Clean test set", "#22c55e"),
        _build_metrics_card("Adversarial Accuracy", f"{adv_acc:.1%}",
                            "Under PGD attack", accent),
        _build_metrics_card("Robust Accuracy", f"{robust_acc:.1%}",
                            "After adversarial retraining", "#636efa"),
        _build_metrics_card("Accuracy Drop", f"{drop:.1%}",
                            "Benign → adversarial", "#ef4444"),
        _build_metrics_card("Circuit Evals", f"{n_q:,}",
                            "Total QPU calls", "#f59e0b"),
        _build_metrics_card("Wall Time", f"{wall_time:.2f}s",
                            "Total runtime", "#8b5cf6"),
    ])

    dataset_badge = _build_dataset_badge(dataset_info)
    config_badge = _build_config_badge(config)

    chart_html = _build_accuracy_chart(records)
    baseline_section = _build_baseline_section(benign_acc, adv_acc, baselines)

    phases = []
    for r in records:
        p = r.get("phase", "?")
        if not phases or phases[-1] != p:
            phases.append(p)
    phases_html = " → ".join(
        f'<span style="background:{accent}22;color:{accent};padding:2px 8px;'
        f'border-radius:4px;font-size:0.8rem;">{p}</span>'
        for p in phases
    )

    table_rows = "".join(
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
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adversarial Pipeline Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        background:#f8fafc;color:#1e293b;line-height:1.5; }}
.container {{ max-width:1100px;margin:0 auto;padding:24px; }}
h1 {{ font-size:1.6rem;font-weight:700;margin-bottom:4px; }}
.subtitle {{ color:#64748b;font-size:0.9rem;margin-bottom:24px; }}
.metrics-grid {{ display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px; }}
code {{ background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:0.85em; }}
table {{ width:100%;border-collapse:collapse; }}
th, td {{ padding:8px;text-align:left;border-bottom:1px solid #e2e8f0; }}
th {{ background:#f1f5f9;font-weight:600;font-size:0.85rem; }}
tr:hover {{ background:#f8fafc; }}
</style>
</head>
<body>
<div class="container">
  <h1>Adversarial Pipeline Report</h1>
  <p class="subtitle">
    PennyLane adversarial attack pipeline —
    PGD attack, adversarial retraining, and robustness evaluation.
  </p>

  <div class="metrics-grid">{cards}</div>

  {dataset_badge}
  {config_badge}

  <div style="margin-bottom:12px;">
    <strong style="color:#475569;font-size:0.8rem;">Pipeline phases:</strong>
    <div style="margin-top:6px;">{phases_html}</div>
  </div>

  <div style="margin-bottom:24px;">
    {chart_html}
  </div>
  {baseline_section}

  <details style="margin-bottom:24px;">
    <summary style="cursor:pointer;color:{accent};font-weight:500;font-size:0.85rem;">
      Show raw metrics table
    </summary>
    <table style="width:100%;border-collapse:collapse;font-size:0.75rem;margin-top:8px;">
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
      <tbody>{table_rows}</tbody>
    </table>
  </details>

  <footer style="border-top:1px solid #e2e8f0;padding:24px 0;margin-top:32px;
                 text-align:center;color:#94a3b8;font-size:0.8rem;">
    Generated by pbg-pennylane-adversarial adversarial pipeline run.
  </footer>
</div>
</body>
</html>"""
