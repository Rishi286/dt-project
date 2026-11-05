"""
Statistical analysis of experiment results.
Input:  C:\dt-project\results\summary.csv
Output: C:\dt-project\results\stats_report.txt
        C:\dt-project\results\figures\  (PNG plots)
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, "summary.csv"))

METRICS = {
    "avg_response_time":  "Average Response Time (s)",
    "completion_rate":    "Task Completion Rate",
    "sla_violation_rate": "SLA Violation Rate",
}
ALLOCATORS = ["dt", "threshold", "greedy"]
ALPHA = 0.05

report_lines = []

def log(line=""):
    print(line)
    report_lines.append(line)

# -----------------------------------------------------------------------
# 1. Descriptive statistics
# -----------------------------------------------------------------------
log("=" * 70)
log("DESCRIPTIVE STATISTICS")
log("=" * 70)
for metric, label in METRICS.items():
    log(f"\n{label}")
    desc = df.groupby("allocator")[metric].agg(["mean","std","min","max","count"])
    desc.columns = ["Mean","Std","Min","Max","N"]
    log(desc.round(6).to_string())

# -----------------------------------------------------------------------
# 2. Shapiro-Wilk normality test
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("SHAPIRO-WILK NORMALITY TEST (p < 0.05 -> non-normal)")
log("=" * 70)
normality = {}
for metric in METRICS:
    log(f"\n{metric}")
    normality[metric] = {}
    for alloc in ALLOCATORS:
        vals = df[df["allocator"] == alloc][metric].values
        stat, p = stats.shapiro(vals)
        normal = p >= ALPHA
        normality[metric][alloc] = normal
        log(f"  {alloc:12s}  W={stat:.4f}  p={p:.4f}  {'normal' if normal else 'NON-NORMAL'}")

# -----------------------------------------------------------------------
# 3. Kruskal-Wallis test (non-parametric, safe for small n)
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("KRUSKAL-WALLIS TEST (omnibus, across all 3 allocators)")
log("=" * 70)
kw_results = {}
for metric, label in METRICS.items():
    groups = [df[df["allocator"] == a][metric].values for a in ALLOCATORS]
    stat, p = stats.kruskal(*groups)
    kw_results[metric] = p
    log(f"\n{label}")
    log(f"  H={stat:.4f}  p={p:.6f}  {'SIGNIFICANT' if p < ALPHA else 'not significant'}")

# -----------------------------------------------------------------------
# 4. Pairwise comparisons (Mann-Whitney U + Holm-Bonferroni correction)
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("PAIRWISE MANN-WHITNEY U  (Holm-Bonferroni corrected)")
log("=" * 70)

pairs = list(combinations(ALLOCATORS, 2))

for metric, label in METRICS.items():
    log(f"\n{label}")
    raw_ps = []
    pair_stats = []
    for a1, a2 in pairs:
        v1 = df[df["allocator"] == a1][metric].values
        v2 = df[df["allocator"] == a2][metric].values
        u, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
        raw_ps.append(p)
        pair_stats.append((a1, a2, u, p))

    # Holm-Bonferroni
    k = len(raw_ps)
    order = np.argsort(raw_ps)
    adjusted = np.zeros(k)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(raw_ps[idx] * (k - rank), 1.0)

    for i, (a1, a2, u, p_raw) in enumerate(pair_stats):
        p_adj = adjusted[i]
        sig = "**" if p_adj < ALPHA else "ns"
        log(f"  {a1} vs {a2:12s}  U={u:.1f}  p_raw={p_raw:.4f}  p_adj={p_adj:.4f}  {sig}")

# -----------------------------------------------------------------------
# 5. Effect sizes (Cohen's d)
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("EFFECT SIZES (Cohen's d)  |d|: small=0.2, medium=0.5, large=0.8")
log("=" * 70)

def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

def effect_label(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"

for metric, label in METRICS.items():
    log(f"\n{label}")
    for a1, a2 in pairs:
        v1 = df[df["allocator"] == a1][metric].values
        v2 = df[df["allocator"] == a2][metric].values
        d = cohens_d(v1, v2)
        log(f"  {a1} vs {a2:12s}  d={d:+.4f}  ({effect_label(d)})")

# -----------------------------------------------------------------------
# 6. 95% Confidence intervals (bootstrap)
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("95% CONFIDENCE INTERVALS (bootstrap, n=10000)")
log("=" * 70)

rng = np.random.default_rng(42)

def bootstrap_ci(vals, n=10000):
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])

for metric, label in METRICS.items():
    log(f"\n{label}")
    for alloc in ALLOCATORS:
        vals = df[df["allocator"] == alloc][metric].values
        lo, hi = bootstrap_ci(vals)
        mean = vals.mean()
        log(f"  {alloc:12s}  mean={mean:.6f}  95% CI [{lo:.6f}, {hi:.6f}]")

# -----------------------------------------------------------------------
# 7. Plots
# -----------------------------------------------------------------------
palette = {"dt": "#2563eb", "threshold": "#16a34a", "greedy": "#dc2626"}

for metric, label in METRICS.items():
    fig, ax = plt.subplots(figsize=(7, 4))
    data_plot = [df[df["allocator"] == a][metric].values for a in ALLOCATORS]
    bp = ax.boxplot(data_plot, patch_artist=True, labels=ALLOCATORS)
    for patch, alloc in zip(bp["boxes"], ALLOCATORS):
        patch.set_facecolor(palette[alloc])
        patch.set_alpha(0.7)
    ax.set_title(f"{label} by Allocator")
    ax.set_ylabel(label)
    ax.set_xlabel("Allocator")
    plt.tight_layout()
    fname = os.path.join(FIGURES_DIR, f"{metric}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    log(f"\nSaved: {fname}")

# -----------------------------------------------------------------------
# 8. Per-regime breakdown
# -----------------------------------------------------------------------
log("\n" + "=" * 70)
log("PER-REGIME MEAN RESULTS")
log("=" * 70)
pivot = df.groupby(["allocator","regime"])[list(METRICS.keys())].mean().round(4)
log(pivot.to_string())

# -----------------------------------------------------------------------
# Save report
# -----------------------------------------------------------------------
report_path = os.path.join(RESULTS_DIR, "stats_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

log(f"\nReport saved to {report_path}")

