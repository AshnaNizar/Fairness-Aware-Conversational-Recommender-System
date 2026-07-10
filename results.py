"""
results.py — FA-CRS figures and final tables (Option B two-system version)

Reads outputs/fair/fair_results.json (produced by the Option B fair_rerank.py,
which contains norerank_metrics + fut_curve on the SAME candidates), and
produces the tables/figures for the writeup:

  outputs/figures/results_table.txt      — main comparison table
  outputs/figures/fut_curve_combined.png — main paper figure (FUT curve)
  outputs/figures/bias_comparison.png    — SPD bars: no-rerank vs FA*IR
  outputs/figures/accuracy_comparison.png — NDCG/P/R bars: no-rerank vs FA*IR
  outputs/figures/collapse_curve.png     — collapse rate vs p (honest limitation)

Run:
    python results.py
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── LOAD RESULTS ─────────────────────────────────────────────────────────────

def _load_first(paths, name):
    for p in paths:
        if os.path.exists(p):
            print(f"Loaded {name}: {p}")
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f"{name} not found in any of: {paths}")


fair = _load_first(
    ["outputs/fair/fair_results.json",
     "outputs/kg/fair_results.json",
     "fair_results.json"],
    "fair results"
)

if "norerank_metrics" not in fair:
    raise SystemExit(
        "ERROR: fair_results.json has no 'norerank_metrics' block. "
        "Rerun with the Option B fair_rerank.py that computes the same-run "
        "no-rerank baseline."
    )

norerank  = fair["norerank_metrics"]
fut       = fair["fut_curve"]
p_values  = [r["p"] for r in fut]

# Pick primary_p: the FUT row closest to 0.3 (where the reranker fires but
# collapse is still comparable to the no-rerank baseline).
PRIMARY_P = min(p_values, key=lambda x: abs(x - 0.30))
primary   = next(r for r in fut if r["p"] == PRIMARY_P)


# ─── 1. RESULTS TABLE ─────────────────────────────────────────────────────────

def make_table():
    lines = []
    lines.append("=" * 76)
    lines.append("TABLE 1: LightGCN vs LightGCN+FA*IR")
    lines.append("Same run, same candidates, identical test split.")
    lines.append(f"Primary operating point: p={PRIMARY_P:.2f}")
    lines.append("=" * 76)
    lines.append(f"{'Metric':<26} {'No-Rerank':>12} {'FA*IR':>12} {'Delta':>12}")
    lines.append("-" * 76)

    rows = [
        ("NDCG@10",              "ndcg_at_10"),
        ("Precision@10",         "precision_at_10"),
        ("Recall@10",            "recall_at_10"),
        ("Gender SPD",           "gender_spd"),
        ("Gender EOD",           "gender_eod"),
        ("Region SPD",           "region_spd"),
        ("Region EOD",           "region_eod"),
        ("Gender rND",           "gender_rND"),
        ("Gender Exposure Gap",  "gender_exposure_gap"),
        ("Gender Collapse Rate", "gender_collapse_rate"),
        ("Region rND",           "region_rND"),
        ("Region Exposure Gap",  "region_exposure_gap"),
        ("Region Collapse Rate", "region_collapse_rate"),
        ("Gini Exposure",        "gini_exposure"),
        ("Catalog Coverage",     "catalog_coverage"),
    ]
    for label, key in rows:
        nb = norerank.get(key)
        fv = primary.get(key)
        if nb is None or fv is None:
            continue
        d = fv - nb
        lines.append(f"{label:<26} {nb:>12.4f} {fv:>12.4f} {d:>+12.4f}")

    lines.append("=" * 76)
    lines.append("")
    lines.append("Reading:")
    g_d = primary["gender_spd"] - norerank["gender_spd"]
    r_d = primary["region_spd"] - norerank["region_spd"]
    n_d = primary["ndcg_at_10"] - norerank["ndcg_at_10"]
    lines.append(f"  Gender SPD change: {g_d:+.4f}  (positive = bias reduced)")
    lines.append(f"  Region SPD change: {r_d:+.4f}  (positive = bias reduced)")
    lines.append(f"  NDCG cost:         {n_d:+.4f}")
    lines.append("")
    lines.append("Note: Collapse Rate rises with p — see collapse_curve.png.")

    s = "\n".join(lines)
    print(s)
    with open(os.path.join(OUTPUT_DIR, "results_table.txt"), "w", encoding="utf-8") as f:
        f.write(s)
    print(f"\nSaved: results_table.txt")


# ─── 2. FUT CURVE ─────────────────────────────────────────────────────────────

def plot_fut_combined():
    ndcg_vals  = [r["ndcg_at_10"] for r in fut]
    gender_spd = [abs(r["gender_spd"]) for r in fut]
    region_spd = [abs(r["region_spd"]) for r in fut]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    c_ndcg, c_g, c_r = "#2563EB", "#DC2626", "#16A34A"

    ax1.plot(p_values, ndcg_vals, "o-", color=c_ndcg, lw=2, ms=6,
             label="NDCG@10 (accuracy)")
    ax1.set_xlabel("Fairness Constraint Strength (p)", fontsize=11)
    ax1.set_ylabel("NDCG@10", fontsize=11, color=c_ndcg)
    ax1.tick_params(axis="y", labelcolor=c_ndcg)
    ax1.set_ylim(0, max(ndcg_vals) * 1.4)

    ax2 = ax1.twinx()
    ax2.plot(p_values, gender_spd, "s--", color=c_g, lw=2, ms=6, label="|Gender SPD|")
    ax2.plot(p_values, region_spd, "^--", color=c_r, lw=2, ms=6, label="|Region SPD|")
    ax2.set_ylabel("|SPD| (lower = fairer)", fontsize=11)
    ax2.set_ylim(0, 1.1)

    ax2.axhline(abs(norerank["gender_spd"]), color=c_g, ls=":", lw=1, alpha=0.6)
    ax2.axhline(abs(norerank["region_spd"]), color=c_r, ls=":", lw=1, alpha=0.6)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, framealpha=0.9)

    plt.title("Fairness-Utility Tradeoff (FUT) Curve\n"
              "Larger p -> stronger fairness constraint, lower NDCG",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fut_curve_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: fut_curve_combined.png")


# ─── 3. BIAS COMPARISON ───────────────────────────────────────────────────────

def plot_bias_comparison():
    systems = ["No-Rerank", f"FA*IR (p={PRIMARY_P:.2f})"]
    g = [abs(norerank["gender_spd"]), abs(primary["gender_spd"])]
    r = [abs(norerank["region_spd"]), abs(primary["region_spd"])]

    x = np.arange(len(systems)); w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax.bar(x - w/2, g, w, label="|Gender SPD|", color="#DC2626", alpha=0.85)
    b2 = ax.bar(x + w/2, r, w, label="|Region SPD|", color="#16A34A", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("|SPD| (lower = fairer)", fontsize=11)
    ax.set_title(f"Bias Reduction: No-Rerank vs FA*IR (p={PRIMARY_P:.2f})\n"
                 "Same run — only difference is reranking",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "bias_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: bias_comparison.png")


# ─── 4. ACCURACY COMPARISON ───────────────────────────────────────────────────

def plot_accuracy_comparison():
    systems = ["No-Rerank", f"FA*IR (p={PRIMARY_P:.2f})"]
    ndcg = [norerank["ndcg_at_10"], primary["ndcg_at_10"]]
    prec = [norerank["precision_at_10"], primary["precision_at_10"]]
    rec  = [norerank["recall_at_10"], primary["recall_at_10"]]

    x = np.arange(len(systems)); w = 0.25
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(x - w, ndcg, w, label="NDCG@10", color="#2563EB", alpha=0.85)
    ax.bar(x,      prec, w, label="Precision@10", color="#7C3AED", alpha=0.85)
    ax.bar(x + w,  rec,  w, label="Recall@10", color="#0891B2", alpha=0.85)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Accuracy Cost of FA*IR Reranking", fontsize=11, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(systems, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: accuracy_comparison.png")


# ─── 5. COLLAPSE RATE CURVE ───────────────────────────────────────────────────

def plot_collapse_curve():
    g_col = [r["gender_collapse_rate"] for r in fut]
    r_col = [r["region_collapse_rate"] for r in fut]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(p_values, g_col, "s-", color="#DC2626", lw=2, ms=6,
            label="Gender Collapse Rate")
    ax.plot(p_values, r_col, "^-", color="#16A34A", lw=2, ms=6,
            label="Region Collapse Rate")
    ax.axhline(norerank["gender_collapse_rate"], color="#DC2626", ls=":", lw=1, alpha=0.5)
    ax.axhline(norerank["region_collapse_rate"], color="#16A34A", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Fairness Constraint Strength (p)", fontsize=11)
    ax.set_ylabel("Collapse Rate (lower = more diverse)", fontsize=11)
    ax.set_title("Collapse Rate vs p — the scarce-supply limitation",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "collapse_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: collapse_curve.png")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...\n")
    make_table()
    print()
    plot_fut_combined()
    plot_bias_comparison()
    plot_accuracy_comparison()
    plot_collapse_curve()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
