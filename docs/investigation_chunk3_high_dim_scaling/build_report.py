"""Builds the chunk-3 HTML investigation report from results.json (produced by
run_sweep.py). Every number quoted in the report text is computed here from
that JSON -- nothing is hand-typed from memory of earlier terminal output.

Run from the repo root: uv run python docs/investigation_chunk3_high_dim_scaling/build_report.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
RESULTS_PATH = _HERE / "results.json"
OUT_PATH = _HERE / "report.html"

with open(RESULTS_PATH) as f:
    R = json.load(f)

# validated reference categorical palette (dataviz skill, palette.md) -- slots 1 & 2
BLUE = "#2a78d6"
AQUA = "#1baf7a"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"

BASE_LAYOUT = dict(
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_SECONDARY, size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    yaxis=dict(gridcolor=GRID, gridwidth=1, zeroline=False, range=[0, 1.05],
               tickformat=".0%", title="Benign accuracy"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def xaxis_style(**overrides):
    return dict(gridcolor=GRID, gridwidth=1, zeroline=False, **overrides)


def mean(xs):
    return sum(xs) / len(xs)


def fmt_pct(x):
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# Experiment 1: position sensitivity
# ---------------------------------------------------------------------------
exp1 = R["position_sensitivity"]
first_accs = [r["accuracy"] for r in exp1 if r["position"] == "first"]
last_accs = [r["accuracy"] for r in exp1 if r["position"] == "last"]

fig1 = go.Figure()
for position, accs, color, xoff in [("first column", first_accs, BLUE, -0.15), ("last column", last_accs, AQUA, 0.15)]:
    seeds = [r["seed"] for r in exp1 if r["position"] == position.split()[0]]
    fig1.add_trace(go.Scatter(
        x=[xoff + 0.02 * (s - 3) for s in seeds], y=accs,
        mode="markers", name=f"{position} (per seed)",
        marker=dict(size=10, color=color, opacity=0.55, line=dict(width=2, color=SURFACE)),
        hovertemplate=f"{position}, seed %{{text}}<br>accuracy=%{{y:.1%}}<extra></extra>",
        text=seeds,
        showlegend=True,
    ))
    fig1.add_trace(go.Scatter(
        x=[xoff], y=[mean(accs)], mode="markers",
        marker=dict(size=16, color=color, symbol="diamond", line=dict(width=2, color=SURFACE)),
        name=f"{position} mean", hovertemplate=f"{position} mean=%{{y:.1%}}<extra></extra>",
        showlegend=False,
    ))
fig1.update_layout(
    **BASE_LAYOUT,
    title="Same input_dim=16 (no truncation either way) -- does column position matter?",
    xaxis=xaxis_style(tickvals=[-0.15, 0.15], ticktext=["first column", "last column"], range=[-0.4, 0.4]),
    height=420,
)

# ---------------------------------------------------------------------------
# Experiment 2: redundancy sweep
# ---------------------------------------------------------------------------
exp2 = R["redundancy_sweep"]
meta2 = R["redundancy_sweep_meta"]
dims = sorted(set(r["input_dim"] for r in exp2))

fig2 = go.Figure()
for variant, label, color in [
    ("original_no_fix", "pre-fix code (no mitigation)", BLUE),
    ("current_pca_fix", "current code (PCA mitigation)", AQUA),
]:
    means = []
    for d in dims:
        pts = [r["accuracy"] for r in exp2 if r["input_dim"] == d and r["variant"] == variant]
        means.append(mean(pts))
        fig2.add_trace(go.Scatter(
            x=[d] * len(pts), y=pts, mode="markers",
            marker=dict(size=8, color=color, opacity=0.45, line=dict(width=1, color=SURFACE)),
            showlegend=False, hovertemplate=f"{label}<br>input_dim=%{{x}}<br>accuracy=%{{y:.1%}}<extra></extra>",
        ))
    fig2.add_trace(go.Scatter(
        x=dims, y=means, mode="lines+markers", name=label,
        line=dict(color=color, width=2), marker=dict(size=9, color=color, line=dict(width=2, color=SURFACE)),
        hovertemplate=f"{label}<br>input_dim=%{{x}}<br>mean accuracy=%{{y:.1%}}<extra></extra>",
    ))

fig2.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"), annotation_text="chance (2-class)",
               annotation_position="bottom right", annotation_font=dict(color=MUTED, size=11))
fig2.add_vline(x=meta2["current_pca_trigger_dim"], line=dict(color=AQUA, width=1, dash="dash"),
               annotation_text=f"PCA triggers  (dim>{meta2['current_pca_trigger_dim']})",
               annotation_position="top", annotation_font=dict(color=AQUA, size=10))
fig2.add_vline(x=meta2["original_truncation_trigger_dim"], line=dict(color=BLUE, width=1, dash="dash"),
               annotation_text=f"truncation starts (dim>{meta2['original_truncation_trigger_dim']})",
               annotation_position="top", annotation_font=dict(color=BLUE, size=10))
fig2.update_layout(
    **BASE_LAYOUT,
    title="Accuracy vs. input_dim, fixed circuit (num_qubits=4, num_layers=4) -- pre-fix vs. current code",
    xaxis=xaxis_style(title="input_dim", tickvals=dims),
    height=460,
)

# ---------------------------------------------------------------------------
# Experiment 3: rejected fix -- layers+epochs scaling
# ---------------------------------------------------------------------------
exp3 = R["layers_epoch_scaling"]
epochs_list = sorted(set(r["epochs"] for r in exp3))

fig3 = go.Figure()
means3 = []
for e in epochs_list:
    pts = [r["accuracy"] for r in exp3 if r["epochs"] == e]
    means3.append(mean(pts))
    fig3.add_trace(go.Scatter(
        x=[e] * len(pts), y=pts, mode="markers",
        marker=dict(size=9, color=BLUE, opacity=0.5, line=dict(width=1, color=SURFACE)),
        showlegend=False, hovertemplate=f"epochs=%{{x}}<br>accuracy=%{{y:.1%}}<extra></extra>",
    ))
fig3.add_trace(go.Scatter(
    x=epochs_list, y=means3, mode="lines+markers", name="num_layers=8 (forced), mean",
    line=dict(color=BLUE, width=2), marker=dict(size=10, color=BLUE, line=dict(width=2, color=SURFACE)),
    hovertemplate="epochs=%{x}<br>mean accuracy=%{y:.1%}<extra></extra>",
))
fig3.add_hline(y=0.5, line=dict(color=MUTED, width=1, dash="dot"), annotation_text="chance (2-class)",
               annotation_position="bottom right", annotation_font=dict(color=MUTED, size=11))
fig3.update_layout(
    **BASE_LAYOUT,
    title="input_dim=48, num_layers manually raised to 8 -- more training epochs does not rescue it",
    xaxis=xaxis_style(title="training_epochs", tickvals=epochs_list),
    height=420,
    showlegend=False,
)

# ---------------------------------------------------------------------------
# Computed figures for the narrative text (all derived from R, not memorized)
# ---------------------------------------------------------------------------
first_mean, last_mean = mean(first_accs), mean(last_accs)
dim8_pooled = [r["accuracy"] for r in exp2 if r["input_dim"] == 8]
dim48_orig = [r["accuracy"] for r in exp2 if r["input_dim"] == 48 and r["variant"] == "original_no_fix"]
dim48_fix = [r["accuracy"] for r in exp2 if r["input_dim"] == 48 and r["variant"] == "current_pca_fix"]
dim24_orig = [r["accuracy"] for r in exp2 if r["input_dim"] == 24 and r["variant"] == "original_no_fix"]
dim24_fix = [r["accuracy"] for r in exp2 if r["input_dim"] == 24 and r["variant"] == "current_pca_fix"]
exp3_e8 = [r["accuracy"] for r in exp3 if r["epochs"] == 8]
exp3_e50 = [r["accuracy"] for r in exp3 if r["epochs"] == max(epochs_list)]

seeds1 = sorted(set(r["seed"] for r in exp1))
seeds2 = sorted(set(r["seed"] for r in exp2))
seeds3 = sorted(set(r["seed"] for r in exp3))

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Chunk 3 Investigation: High-Dimensional Feature Scaling</title>
<style>
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; color: {INK};
         background: #f9f9f7; margin: 0; padding: 0 0 60px 0; }}
  .wrap {{ max-width: 920px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin-top: 48px; border-top: 1px solid {GRID}; padding-top: 24px; }}
  .subtitle {{ color: {INK_SECONDARY}; font-size: 0.95rem; margin-bottom: 24px; }}
  .callout {{ background: {BLUE}0d; border-left: 4px solid {BLUE}; border-radius: 8px;
              padding: 16px 20px; margin: 16px 0; font-size: 0.9rem; }}
  .caveat {{ background: #eda1000d; border-left: 4px solid #eda100; border-radius: 8px;
             padding: 16px 20px; margin: 16px 0; font-size: 0.9rem; }}
  .chart-card {{ background: {SURFACE}; border: 1px solid {GRID}; border-radius: 12px;
                 padding: 16px; margin: 16px 0; }}
  table.data {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 8px; }}
  table.data th, table.data td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid {GRID}; }}
  table.data th:first-child, table.data td:first-child {{ text-align: left; }}
  code {{ background: #f0efec; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid {GRID};
            font-size: 0.78rem; color: {MUTED}; }}
</style>
</head>
<body>
<div class="wrap">

<h1>High-Dimensional Feature Scaling in <code>PennyLaneAdversarialProcess</code></h1>
<div class="subtitle">
  Investigation backing NEXT_STEPS.md gap 3 (<code>I=1, E=2, S=1 &rarr; 2</code>) &mdash;
  every number and chart below is computed from a real, reproducible sweep of
  {len(exp1) + len(exp2) + len(exp3)} live PennyLane <code>lightning.qubit</code> circuit
  trainings (no mocked data), run in {R['meta']['wall_time_s']:.0f}s.
</div>

<div class="callout">
  <strong>Why this exists:</strong> the plan explicitly deferred gap 3 &mdash;
  &ldquo;only act if &gt;50 features hurt accuracy&rdquo; &mdash; and flagged two candidate
  fixes without picking one. Before writing code, the trigger condition itself needed
  checking; after writing code, the first candidate fix (auto-raising <code>num_layers</code>)
  turned out not to work and was replaced. This report is the evidence trail for both calls.
</div>

<h2>1. Does circuit position matter, independent of dimensionality?</h2>
<p>
  At <code>input_dim=16</code> with <code>num_qubits=4, num_layers=4</code>
  (<code>weights_elements=48</code>), <code>num_reup=3</code> &mdash; well above the
  redundancy floor, so neither the pre-fix truncation bug nor the current PCA mitigation
  ever triggers here. The <em>only</em> thing that differs between the two groups below is
  which column (0 or 15) carries the discriminative signal.
</p>
<div class="chart-card">{fig1.to_html(include_plotlyjs="cdn", full_html=False, div_id="fig1")}</div>
<p>
  Mean accuracy: <strong>first column {fmt_pct(first_mean)}</strong> (seeds {seeds1}: {[round(a,3) for a in first_accs]})
  vs. <strong>last column {fmt_pct(last_mean)}</strong> ({[round(a,3) for a in last_accs]}).
  <code>forward()</code> reshapes the (tiled) input vector directly into the
  <code>(n_layers, n_wires, 3)</code> weight tensor via <code>torch.reshape</code>, so a
  feature's column index fixes which specific rotation gate it multiplies into. Some
  positions sit on a more favorable gradient landscape at random init than others &mdash;
  a real effect, and one that no dimensionality-reduction step can undo, because PCA
  components still have to land <em>somewhere</em> in that same reshape.
</p>

<h2>2. Does accuracy actually degrade with input_dim, and does the PCA fix help?</h2>
<p>
  Sweeping <code>input_dim</code> from {min(dims)} to {max(dims)} on the same fixed circuit,
  comparing the genuine pre-fix code (<code>git show HEAD:pbg_pennylane_adversarial/processes.py</code>,
  no mitigation) against the current code (PCA mitigation, <code>MIN_NUM_REUP={meta2['current_min_num_reup']}</code>).
  {len(seeds2)} seeds per point ({seeds2}).
</p>
<div class="chart-card">{fig2.to_html(include_plotlyjs=False, full_html=False, div_id="fig2")}</div>
<p>
  The trigger condition holds: at <code>input_dim=8</code> both variants average
  {fmt_pct(mean(dim8_pooled))} (well-conditioned, no mitigation needed by either), but by
  <code>input_dim=48</code> the pre-fix code averages {fmt_pct(mean(dim48_orig))} and the
  PCA-fixed code averages {fmt_pct(mean(dim48_fix))} &mdash; both near chance, and
  <strong>not meaningfully different from each other at this dimension</strong>. At
  <code>input_dim=24</code> &mdash; exactly where the current code's PCA trigger sits &mdash;
  the two variants are identical ({fmt_pct(mean(dim24_orig))} vs. {fmt_pct(mean(dim24_fix))})
  because PCA hasn't kicked in yet (<code>24</code> is the threshold, not yet exceeded).
  Looking across the full sweep, the PCA-fixed line sits at or slightly above the pre-fix
  line at most dimensions, but the gap is small relative to the seed-to-seed spread
  (visible as the scattered dots behind each mean line) &mdash; PCA is not pulling accuracy
  back up to the <code>input_dim=8</code> ceiling.
</p>

<h2>3. The rejected fix: does more training rescue a bigger circuit?</h2>
<p>
  The plan's first suggested fix was auto-raising <code>num_layers</code> to restore
  <code>num_reup</code>. Manually forcing <code>num_layers=8</code> at
  <code>input_dim=48</code> (which raises <code>weights_elements</code> to 96, restoring
  <code>num_reup=2</code>) and sweeping the training budget from {min(epochs_list)} to
  {max(epochs_list)} epochs, {len(seeds3)} seeds per point ({seeds3}):
</p>
<div class="chart-card">{fig3.to_html(include_plotlyjs=False, full_html=False, div_id="fig3")}</div>
<p>
  Mean accuracy at {min(epochs_list)} epochs: {fmt_pct(mean(exp3_e8))}. At
  {max(epochs_list)} epochs &mdash; a {max(epochs_list)//min(epochs_list)}x larger training
  budget &mdash; mean accuracy: {fmt_pct(mean(exp3_e50))}. No recovery. This is why the
  auto-raise-<code>num_layers</code> approach was dropped in favor of PCA: growing the
  circuit doesn't just cost more compute, it doesn't converge to a better answer even when
  given far more of that compute, consistent with a barren-plateau-type effect in deeper
  <code>StronglyEntanglingLayers</code> circuits.
</p>

<h2>Synthesis</h2>
<p>
  Three independent effects are visible in this data, and only one of them is addressed by
  the shipped fix:
</p>
<ul>
  <li><strong>Redundancy collapse</strong> (<code>num_reup</code> &le; 1) &mdash; the effect
    the plan named. PCA addresses this directly by keeping <code>input_dim</code> within the
    circuit's capacity.</li>
  <li><strong>Position sensitivity</strong> (Section 1) &mdash; not addressed by PCA or any
    fix implemented here. A component that captures the discriminative signal can still land
    on an unfavorable reshape position.</li>
  <li><strong>Circuit-depth trainability</strong> (Section 3) &mdash; the reason growing the
    circuit was rejected as a fix; PCA avoids this by construction (it never changes
    <code>num_qubits</code>/<code>num_layers</code>), but doesn't reverse it either.</li>
</ul>

<div class="caveat">
  <strong>Honest caveat.</strong> Section 2's own data shows the PCA fix's effect size is
  small relative to seed-to-seed noise at the dimensions where it's needed most
  (<code>input_dim=48</code>: {fmt_pct(mean(dim48_orig))} unfixed vs.
  {fmt_pct(mean(dim48_fix))} fixed). It guarantees the necessary mathematical condition
  (<code>num_reup &ge; MIN_NUM_REUP</code>) but not the sufficient one (good accuracy),
  because Sections 1 and 3's effects are untouched by it. This is harm reduction on a
  genuinely hard optimization problem, appropriately scoped to a low-priority
  (<code>I=1, E=2, S=1</code>) chunk &mdash; not a claim that high-dimensional datasets will
  now train well.
</div>

<h2>Raw data</h2>
<details>
  <summary style="cursor:pointer;color:{INK_SECONDARY};">Experiment 1 &mdash; position sensitivity</summary>
  <table class="data">
    <tr><th>position</th><th>seed</th><th>accuracy</th></tr>
    {"".join(f"<tr><td>{r['position']}</td><td>{r['seed']}</td><td>{r['accuracy']:.3f}</td></tr>" for r in exp1)}
  </table>
</details>
<details style="margin-top:8px;">
  <summary style="cursor:pointer;color:{INK_SECONDARY};">Experiment 2 &mdash; redundancy sweep</summary>
  <table class="data">
    <tr><th>input_dim</th><th>variant</th><th>seed</th><th>accuracy</th></tr>
    {"".join(f"<tr><td>{r['input_dim']}</td><td>{r['variant']}</td><td>{r['seed']}</td><td>{r['accuracy']:.3f}</td></tr>" for r in exp2)}
  </table>
</details>
<details style="margin-top:8px;">
  <summary style="cursor:pointer;color:{INK_SECONDARY};">Experiment 3 &mdash; layers+epochs scaling</summary>
  <table class="data">
    <tr><th>epochs</th><th>seed</th><th>accuracy</th></tr>
    {"".join(f"<tr><td>{r['epochs']}</td><td>{r['seed']}</td><td>{r['accuracy']:.3f}</td></tr>" for r in exp3)}
  </table>
</details>

<footer>
  Generated from <code>chunk3_results.json</code>, produced by a standalone sweep script
  (not part of the installable package) that instantiates <code>PennyLaneAdversarialProcess</code>
  directly &mdash; both the current working-tree version and the pre-fix version loaded via
  <code>git show HEAD:pbg_pennylane_adversarial/processes.py</code> &mdash; and drives
  <code>update()</code> through <code>init &rarr; training &rarr; benign_eval</code> on
  synthetic classification data. All circuit training is real (PennyLane <code>lightning.qubit</code>
  + PyTorch autograd), not mocked. Total wall time: {R['meta']['wall_time_s']:.1f}s across
  {len(exp1) + len(exp2) + len(exp3)} runs.
</footer>

</div>
</body>
</html>
"""

with open(OUT_PATH, "w") as f:
    f.write(html)
print(f"Wrote {OUT_PATH}")
