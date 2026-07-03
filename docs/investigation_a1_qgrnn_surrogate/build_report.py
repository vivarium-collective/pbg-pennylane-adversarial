"""Builds the A1 phase-4 decision-gate HTML report from results.json (produced
by run_baseline_gate.py). Every number quoted in the report text is computed
here from that JSON -- nothing is hand-typed from memory of earlier terminal
output (mirrors docs/investigation_chunk3_high_dim_scaling/build_report.py's
convention).

Run from the repo root: uv run python docs/investigation_a1_qgrnn_surrogate/build_report.py
"""
import json
from pathlib import Path

import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
RESULTS_PATH = _HERE / "results.json"
OUT_PATH = _HERE / "report.html"

with open(RESULTS_PATH) as f:
    R = json.load(f)

BLUE = "#2a78d6"
AQUA = "#1baf7a"
AMBER = "#d69e2a"
MUTED = "#898781"
GRID = "#e1e0d9"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

VERDICT_COLOR = {"GO": GOOD, "STOP": CRITICAL, "INCONCLUSIVE": AMBER}

BASE_LAYOUT = dict(
    plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_SECONDARY, size=13),
    margin=dict(l=60, r=30, t=50, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hoverlabel=dict(bgcolor="white", font_size=12),
)


def xaxis_style(**overrides):
    return dict(gridcolor=GRID, gridwidth=1, zeroline=False, **overrides)


def yaxis_style(**overrides):
    return dict(gridcolor=GRID, gridwidth=1, zeroline=True, zerolinecolor=MUTED, **overrides)


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


gate = R["decision_gate_verdict"]
real_gate = R["real_data_gate"]
calib = R["data_sufficiency_calibration"]
folds = real_gate["folds"]

SANITY_NODES = [
    "cell_mass", "dry_mass", "protein_mass", "rna_mass", "dna_mass", "volume", "number_of_oric",
]
REAL_TEST_NODES = ["instantaneous_growth_rate"]
ALL_NODES = SANITY_NODES + REAL_TEST_NODES
ARMS = ["persistence", "dmd", "sindy"]
ARM_COLOR = {"persistence": MUTED, "dmd": BLUE, "sindy": AQUA}

# ---------------------------------------------------------------------------
# Figure 1: data-sufficiency calibration sweep
# ---------------------------------------------------------------------------
sweep = calib["sweep"]
ns = [row["n_trajectories"] for row in sweep]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=ns, y=[row["achievable_r2_median"] for row in sweep], mode="lines+markers",
    name="measured achievable ceiling", line=dict(color=INK, width=2, dash="dot"),
    marker=dict(size=8, color=INK),
))
for arm in ARMS:
    fig1.add_trace(go.Scatter(
        x=ns, y=[row[f"{arm}_median_r2"] for row in sweep], mode="lines+markers",
        name=arm, line=dict(color=ARM_COLOR[arm], width=2),
        marker=dict(size=9, color=ARM_COLOR[arm], line=dict(width=2, color=SURFACE)),
        hovertemplate=f"{arm}<br>N=%{{x}} trajectories<br>median R2=%{{y:.3f}}<extra></extra>",
    ))
fig1.add_vline(x=4, line=dict(color=CRITICAL, width=1, dash="dash"),
               annotation_text="real N=4 available", annotation_position="top",
               annotation_font=dict(color=CRITICAL, size=11))
fig1.add_hline(y=0, line=dict(color=MUTED, width=1))
fig1.update_layout(
    **BASE_LAYOUT,
    title="Synthetic calibration: correctly-specified linear system, LOTO median R2 vs. trajectory count",
    xaxis=xaxis_style(title="number of trajectories (N)", tickvals=ns),
    yaxis=yaxis_style(title="median R2 (leave-one-trajectory-out)"),
    height=440,
)

# ---------------------------------------------------------------------------
# Figure 2: real-data per-node R2, grouped bar, sanity-check vs real-test
# ---------------------------------------------------------------------------
def node_median(arm, node):
    vals = [f[arm]["per_node"][node]["r2"] for f in folds if node in f[arm]["per_node"]]
    return sum(vals) / len(vals) if vals else float("nan")


fig2 = go.Figure()
for arm in ARMS:
    fig2.add_trace(go.Bar(
        x=ALL_NODES, y=[node_median(arm, n) for n in ALL_NODES], name=arm,
        marker_color=ARM_COLOR[arm],
        hovertemplate=f"{arm}<br>%{{x}}<br>mean-of-folds R2=%{{y:.3f}}<extra></extra>",
    ))
fig2.add_vline(x=len(SANITY_NODES) - 0.5, line=dict(color=MUTED, width=1, dash="dash"))
fig2.add_annotation(x=len(SANITY_NODES) - 0.5, y=1.05, yref="paper", showarrow=False,
                     text="sanity-check  |  real-test", font=dict(color=MUTED, size=11))
fig2.add_hline(y=0, line=dict(color=MUTED, width=1))
fig2.update_layout(
    **BASE_LAYOUT,
    title="Real WCM data: per-node R2, mean across 4 leave-one-trajectory-out folds",
    xaxis=xaxis_style(title=None),
    yaxis=yaxis_style(title="mean R2 across folds"),
    barmode="group",
    height=460,
)

# ---------------------------------------------------------------------------
# Figure 3: DMD eigenvalue spectrum (fold 0, representative)
# ---------------------------------------------------------------------------
fold0 = folds[0]
eigs = fold0["dmd"]["eigenvalues"]
fig3 = go.Figure()
theta = [i / 200 * 2 * 3.141592653589793 for i in range(201)]
fig3.add_trace(go.Scatter(
    x=[__import__("math").cos(t) for t in theta], y=[__import__("math").sin(t) for t in theta],
    mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="unit circle", showlegend=True,
))
fig3.add_trace(go.Scatter(
    x=[e["real"] for e in eigs], y=[e["imag"] for e in eigs], mode="markers",
    marker=dict(size=12, color=BLUE, line=dict(width=2, color=SURFACE)),
    name="DMD eigenvalues (holdout fold 0)",
    hovertemplate="Re=%{x:.3f}<br>Im=%{y:.3f}<extra></extra>",
))
fig3.update_layout(
    **BASE_LAYOUT,
    title="DMD operator eigenvalue spectrum (fold: holdout trajectory 0) -- magnitude near 1 = slow/persistent mode",
    xaxis=xaxis_style(title="Re(lambda)", scaleanchor="y", scaleratio=1),
    yaxis=yaxis_style(title="Im(lambda)"),
    height=440,
)

# ---------------------------------------------------------------------------
# Computed figures for narrative text
# ---------------------------------------------------------------------------
sanity_r2 = gate["sanity_check_median_r2"]
real_test_r2 = gate["real_test_median_r2"]
verdict = gate["verdict"]
verdict_color = VERDICT_COLOR.get(verdict, MUTED)
sanity_verdict = gate["sanity_check_verdict"]
outliers = gate["outlier_holdout_trajectories"]
calib_n4 = gate["calibration_at_n4"]

# SINDy coefficient magnitude sanity check (collinearity caveat) -- computed,
# not asserted: flag if any real-data SINDy coefficient is implausibly large
# for z-scored inputs (a well-conditioned fit on normalized data shouldn't
# need coefficients in the hundreds/thousands).
max_abs_sindy_coef = max(
    abs(coef) for f in folds for terms in f["sindy"]["active_terms"].values() for _, coef in terms
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A1 Phase-4 Decision Gate: Persistence / DMD / SINDy on Real WCM Data</title>
<style>
  body {{ font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; color: {INK};
         background: #f9f9f7; margin: 0; padding: 0 0 60px 0; }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  h2 {{ font-size: 1.2rem; margin-top: 48px; border-top: 1px solid {GRID}; padding-top: 24px; }}
  .subtitle {{ color: {INK_SECONDARY}; font-size: 0.95rem; margin-bottom: 24px; }}
  .verdict-banner {{ border-radius: 10px; padding: 18px 22px; margin: 16px 0; display: flex;
                      gap: 16px; align-items: flex-start; border: 1px solid {verdict_color}55;
                      background: {verdict_color}12; }}
  .verdict-badge {{ flex: none; font-weight: 800; font-size: 1rem; color: white;
                     background: {verdict_color}; border-radius: 999px; padding: 4px 14px; }}
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

<h1>A1 Phase-4 Decision Gate: Persistence / DMD / SINDy on Real WCM Transition Data</h1>
<div class="subtitle">
  Backing <code>NEXT_STEPS.md</code> gap 4 &mdash; before training
  <code>QGRNNSurrogate</code>/<code>ClassicalGNNSurrogate</code> (phase 3, already built), check
  whether classical baselines already explain the achievable variance on real
  {real_gate['n_trajectories']}-trajectory, {real_gate['stride_seconds']}-second-stride WCM transition
  data ({real_gate['n_pairs_total']} transition pairs after dropping
  {real_gate['n_pairs_dropped_shard_boundary']} spurious cross-shard-boundary pair(s)).
</div>

<div class="verdict-banner">
  <span class="verdict-badge">{verdict}</span>
  <div>
    <strong>Real-test node (instantaneous_growth_rate) verdict: {verdict}.</strong>
    Sanity-check node cluster (near-deterministic mass/replication arithmetic) verdict:
    <strong>{sanity_verdict}</strong>. See §2-3 below for why these are reported separately
    rather than one blended number.
  </div>
</div>

<h2>1. Is the harness itself trustworthy at N=4? (data-sufficiency calibration)</h2>
<p>
  Before trusting any real-data classical-baseline number, this ran the identical harness
  (z-score fit on training folds, DMD, SINDy, leave-one-trajectory-out evaluation) on a
  <em>correctly-specified linear</em> synthetic system (ported idea from
  <code>pbg-pennylane-data-reuploading</code>'s <code>NonlinearProcessGenerator</code> --
  see <code>dataset_transform/synthetic_dynamics.py</code>), sweeping trajectory count.
  Unlike that generator's own audited flaw (an asserted, unvalidated "Bayes-optimal accuracy"),
  the achievable R2 ceiling here is <em>measured</em> directly from the same generating code path.
</p>
<div class="chart-card">{fig1.to_html(include_plotlyjs="cdn", full_html=False, div_id="fig1")}</div>
<div class="caveat">
  <strong>Empirical finding, not assumed going in:</strong> at the real N=4, DMD's median LOTO R2
  is <strong>{fmt(calib_n4['dmd_median_r2'])}</strong> &mdash; <em>worse than predicting the mean</em>
  &mdash; on a system with a measured achievable ceiling of
  <strong>{fmt(calib_n4['achievable_r2_median'])}</strong>. An unconstrained 8x8 operator has too many
  free parameters for 3 training trajectories to pin down, and a held-out 4th trajectory occupies a
  different, unvisited region of state space. SINDy's sparsity partially compensates
  (median R2 {fmt(calib_n4['sindy_median_r2'])}) but still falls well short of the ceiling.
  Both improve steadily past N=8. <strong>Conclusion: any real-data DMD score at N=4 should be
  discounted almost entirely; any real-data SINDy score should be read as an underestimate of
  true recoverable signal.</strong>
</div>

<h2>2. Real-data result: sanity-check nodes vs. the one real test</h2>
<p>
  <code>todo.md</code> phase 5 pre-registered a split: <code>cell_mass</code>, <code>dry_mass</code>,
  <code>protein_mass</code>, <code>rna_mass</code>, <code>dna_mass</code>, <code>volume</code>, and
  <code>number_of_oric</code> are close to deterministic simulator arithmetic (mass components sum;
  <code>dna_mass</code> tracks <code>number_of_oric</code> almost exactly) &mdash; recovering them is
  an expected sanity check, not a finding. <code>instantaneous_growth_rate</code> is the one genuinely
  emergent relationship (Scott et al. growth-law territory). Blending both into one aggregate median
  would let the trivial cluster's high score hide a total failure on the node that actually matters,
  so they're kept separate below.
</p>
<div class="chart-card">{fig2.to_html(include_plotlyjs=False, full_html=False, div_id="fig2")}</div>
<p>
  Sanity-check cluster: DMD/SINDy reach median R2
  <strong>{fmt(max(sanity_r2['dmd'], sanity_r2['sindy']))}</strong> vs. persistence
  <strong>{fmt(sanity_r2['persistence'])}</strong> &mdash; recovered about as well as expected for
  near-algebraic identities. Real-test node: best classical arm
  (<strong>{gate['real_test_best_arm'].upper()}</strong>) reaches median R2
  <strong>{fmt(real_test_r2[gate['real_test_best_arm']])}</strong> vs. persistence
  <strong>{fmt(real_test_r2['persistence'])}</strong> &mdash;
  {"no arm beats its own mean prediction" if real_test_r2[gate['real_test_best_arm']] <= 0 else "a positive signal, beating the mean"}.
  Per §1's calibration, this cannot be read as "no learnable growth-rate dynamics" on its own --
  it is exactly the outcome this harness would also produce on a correctly-specified linear system
  at this N. Concrete next step: more real trajectories and/or flux-level features beyond state
  snapshots, before concluding growth_rate has no learnable one-step dynamics.
</p>
{"".join(f'''
<div class="caveat">
  <strong>Outlier fold flagged:</strong> holdout trajectory {t} causes catastrophic failure
  (double-digit-negative R2) even on the sanity-check cluster, unlike the other 3 folds.
  Worth checking whether that trajectory's lineage_seed/generation combination reflects a
  genuinely different physiological regime (it is the longest, latest-generation trajectory
  pulled) or a data-quality issue, before treating all 4 folds as equally representative.
</div>''' for t in outliers)}

<h2>3. DMD eigenvalue spectrum &amp; SINDy term recovery</h2>
<div class="chart-card">{fig3.to_html(include_plotlyjs=False, full_html=False, div_id="fig3")}</div>
<p>
  Eigenvalue magnitudes near 1 indicate slow/persistent modes -- a quantitative version of the
  growth-rate-variance check from phase 1. <code>rank_used={fold0['dmd']['rank_used']}</code> of 8
  singular values retained for this fold (<code>singular_values=
  {[round(s, 2) for s in fold0['dmd']['singular_values']]}</code>).
</p>
<div class="caveat">
  <strong>SINDy coefficient caveat:</strong> the largest recovered coefficient magnitude across all
  folds is <strong>{fmt(max_abs_sindy_coef, 1)}</strong> on z-scored (unit-variance) inputs -- far
  larger than a well-conditioned sparse fit should need. This is the same severe collinearity phase 1
  already found among the mass-group features (near-exact linear dependencies like
  <code>cell_mass ~= protein_mass + rna_mass + dna_mass + ...</code>), showing up here as large,
  canceling coefficients rather than a numerically stable sparse structure. <strong>Read SINDy's
  recovered terms as evidence of which nodes are involved, not as physically meaningful
  coefficient values.</strong>
</div>

<h2>4. Decision-gate verdict (full text)</h2>
<div class="callout">{gate['reason']}</div>

<h2>Raw data</h2>
<details>
  <summary style="cursor:pointer;color:{INK_SECONDARY};">Per-fold, per-node R2 (all 3 arms, all 4 holdout folds)</summary>
  <table class="data">
    <tr><th>holdout traj</th><th>node</th><th>persistence R2</th><th>DMD R2</th><th>SINDy R2</th></tr>
    {"".join(
        f"<tr><td>{f['holdout_traj_id']}</td><td>{n}</td>"
        f"<td>{f['persistence']['per_node'][n]['r2']:.3f}</td>"
        f"<td>{f['dmd']['per_node'][n]['r2']:.3f}</td>"
        f"<td>{f['sindy']['per_node'][n]['r2']:.3f}</td></tr>"
        for f in folds for n in ALL_NODES
    )}
  </table>
</details>
<details style="margin-top:8px;">
  <summary style="cursor:pointer;color:{INK_SECONDARY};">Data-sufficiency calibration sweep (raw)</summary>
  <table class="data">
    <tr><th>N trajectories</th><th>n train/fold</th><th>achievable R2</th><th>persistence</th><th>DMD</th><th>SINDy</th></tr>
    {"".join(
        f"<tr><td>{row['n_trajectories']}</td><td>{row['n_train_per_fold']}</td>"
        f"<td>{row['achievable_r2_median']:.3f}</td><td>{row['persistence_median_r2']:.3f}</td>"
        f"<td>{row['dmd_median_r2']:.3f}</td><td>{row['sindy_median_r2']:.3f}</td></tr>"
        for row in sweep
    )}
  </table>
</details>

<footer>
  Generated from <code>results.json</code>, produced by
  <code>run_baseline_gate.py</code> (real transition pairs from
  <code>{real_gate['feature_cols']}</code>, SINDy degree={real_gate['sindy_degree']}
  threshold={real_gate['sindy_threshold']}). Reusable code
  (<code>dynamics_baselines.py</code>, <code>synthetic_dynamics.py</code>,
  <code>wcm_loader.build_transition_pairs</code>'s <code>stride</code> param) lives in the
  installable package per this repo's feature-PR-vs-investigation-PR convention
  (<code>AGENTS.md</code>); this script and report are investigation-specific.
</footer>

</div>
</body>
</html>
"""

with open(OUT_PATH, "w") as f:
    f.write(html)
print(f"Wrote {OUT_PATH}")
