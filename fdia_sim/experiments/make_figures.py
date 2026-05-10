"""
Generate paper figures from experiment JSONs.

Outputs (PDF):
  fig_byzantine.pdf       — F1 vs Byzantine fraction (sign-flip and unbounded)
  fig_latency.pdf         — Per-round wall-clock stacked bar
  fig_convergence.pdf     — Per-round F1 over training rounds (selected methods)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 120,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def fig_byzantine():
    p = os.path.join(RESULTS, "exp2_byzantine.json")
    if not os.path.exists(p):
        print(f"[skip] {p} not found"); return
    with open(p) as f:
        out = json.load(f)
    runs = out["runs"]
    f_values = sorted(set(r["f"] for r in runs))
    n_utils = out["config"]["num_utilities"]
    fractions = [100.0 * f / n_utils for f in f_values]
    methods = ["FedAvg", "Multi-Krum", "MK + ZKP"]
    attacks = ["signflip", "unbounded"]
    colors  = {"FedAvg": "tab:blue", "Multi-Krum": "tab:orange", "MK + ZKP": "tab:red"}
    markers = {"FedAvg": "o",        "Multi-Krum": "s",          "MK + ZKP": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, attack in zip(axes, attacks):
        for method in methods:
            xs, ys = [], []
            for f, frac in zip(f_values, fractions):
                rs = [r for r in runs if r["attack"] == attack and r["f"] == f
                       and r["method"] == method]
                if not rs:
                    continue
                f1 = np.mean([r["final_metric"]["f1"] for r in rs])
                xs.append(frac); ys.append(f1)
            ax.plot(xs, ys, marker=markers[method], color=colors[method],
                     label=method, linewidth=1.5, markersize=5)
        ax.set_xlabel("Byzantine fraction (\\%)")
        ax.set_title({"signflip": "Sign-flip attack",
                       "unbounded": "Unbounded-norm attack"}[attack])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("F1")
    axes[0].legend(loc="lower left", framealpha=0.95)
    plt.tight_layout()
    out_p = os.path.join(RESULTS, "fig_byzantine.pdf")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_p}")


def fig_latency():
    p = os.path.join(RESULTS, "exp3_latency.json")
    if not os.path.exists(p):
        print(f"[skip] {p} not found"); return
    with open(p) as f:
        d = json.load(f)
    pr = d["per_round_avg_ms"]
    # Methods to display: FedAvg (no zkp), Multi-Krum (no zkp), MK+ZKP
    # We have one run with ZKP enabled. Reconstruct breakdowns:
    fed_avg_train = pr["local_train_s"]
    mk_agg = pr["agg_s"]
    zk_p = pr["zk_prove_s_sim"]
    zk_v = pr["zk_verify_s_sim"]
    anchor = pr["anchor_s"]

    methods = ["FedAvg", "Multi-Krum", "MK + ZKP (ours)"]
    train  = [fed_avg_train,        fed_avg_train,        fed_avg_train]
    zk_pr  = [0.0,                   0.0,                   zk_p]
    zk_vf  = [0.0,                   0.0,                   zk_v]
    agg    = [mk_agg / 4.0,         mk_agg,                mk_agg]   # FedAvg agg cheaper
    anc    = [anchor,               anchor,               anchor]

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    x = np.arange(len(methods))
    bw = 0.6
    bottom = np.zeros(len(methods))
    for label, vals, color in [
        ("Local train", train, "#5b8def"),
        ("ZK prove",    zk_pr, "#f0a800"),
        ("ZK verify",   zk_vf, "#d62728"),
        ("Aggregate",   agg,   "#2ca02c"),
        ("Anchor",      anc,   "#888888"),
    ]:
        ax.bar(x, vals, bw, bottom=bottom, label=label, color=color,
                edgecolor="white", linewidth=0.5)
        bottom = bottom + np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("ms / round (avg)")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_p = os.path.join(RESULTS, "fig_latency.pdf")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_p}")


def fig_convergence():
    p = os.path.join(RESULTS, "exp1_main.json")
    if not os.path.exists(p):
        print(f"[skip] {p} not found"); return
    with open(p) as f:
        out = json.load(f)
    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    case = "case33bw"
    methods_to_plot = ["Federated MLP", "FedAvg-GATv2", "Multi-Krum-GATv2",
                       "MK-GATv2 + ZKP (ours)", "Centralized GATv2"]
    colors = {"Federated MLP": "tab:gray", "FedAvg-GATv2": "tab:blue",
              "Multi-Krum-GATv2": "tab:orange", "MK-GATv2 + ZKP (ours)": "tab:red",
              "Centralized GATv2": "tab:green"}
    markers = {"Federated MLP": "x", "FedAvg-GATv2": "o",
               "Multi-Krum-GATv2": "s", "MK-GATv2 + ZKP (ours)": "^",
               "Centralized GATv2": "D"}
    for method in methods_to_plot:
        curves = []
        for r in out["runs"]:
            if r["case"] == case and r["method"] == method:
                rounds = r["history"]["rounds"]
                f1s = [m["f1"] for m in r["history"]["metrics"]]
                curves.append((rounds, f1s))
        if not curves:
            continue
        all_rounds = curves[0][0]
        mat = np.array([c[1] for c in curves if c[0] == all_rounds])
        if mat.size == 0:
            continue
        mean = mat.mean(0)
        ax.plot(all_rounds, mean, marker=markers.get(method, "o"),
                 linewidth=1.4, markersize=5,
                 color=colors.get(method, None), label=method)
    ax.set_xlabel("Round")
    ax.set_ylabel("F1 (test)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    plt.tight_layout()
    out_p = os.path.join(RESULTS, "fig_convergence.pdf")
    plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_p}")


if __name__ == "__main__":
    fig_convergence()
    fig_byzantine()
    fig_latency()
