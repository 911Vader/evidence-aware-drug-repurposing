"""
summarize_results.py

Aggregates the full-scale, multi-seed results/*.json files (3 seeds x 2
splits x 4 variants: baseline, fingerprint, random_fingerprint, adaptive)
into:

  1. results/summary_table.csv       -- mean +/- SD per (split, variant)
  2. figures/comparison_auroc_auprc.png  -- grouped bar chart with error bars
  3. results/evidence_stratification.csv -- cold_start only: compounds
     bucketed into Low/Medium/High non-CtD graph degree (tertiles, computed
     per-seed from that seed's own train-only graph), Hits@10 and mean rank
     per bucket per variant, averaged across seeds.
  4. figures/evidence_stratification.png

Nothing here is estimated -- every number comes from the per_edge
diagnostics and metrics already saved by train.py.

Run:
    python src/summarize_results.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

SPLITS = ["random", "cold_start"]
VARIANTS = ["baseline", "fingerprint", "random_fingerprint", "adaptive"]
# Novelty-control ablation: same gated-fusion architecture as "adaptive",
# fed shuffled fingerprints instead of real ones. Isolates whether
# adaptive's gain is the gating mechanism itself or the real chemistry it
# is given. Kept out of the main 4-way comparison (its own table/chart
# below) so it doesn't crowd the primary result.
ABLATION_VARIANTS = ["adaptive", "adaptive_random_fingerprint"]
SEEDS = [0, 1, 2]
METRICS = ["auroc", "auprc", "hits@1", "hits@3", "hits@10"]


ALL_VARIANTS = VARIANTS + [v for v in ABLATION_VARIANTS if v not in VARIANTS]


def load_all():
    records = defaultdict(dict)  # records[(split,variant)][seed] = full json
    for split in SPLITS:
        for variant in ALL_VARIANTS:
            for seed in SEEDS:
                path = os.path.join("results", f"{split}_{variant}_seed{seed}.json")
                if not os.path.exists(path):
                    print(f"WARNING: missing {path}")
                    continue
                with open(path) as f:
                    records[(split, variant)][seed] = json.load(f)
    return records


def mean_std_table(records):
    rows = []
    print(f"{'run':30s} " + " ".join(f"{m:>14s}" for m in METRICS))
    for split in SPLITS:
        for variant in VARIANTS:
            seed_runs = records[(split, variant)]
            vals = {m: [] for m in METRICS}
            for seed, rec in seed_runs.items():
                for m in METRICS:
                    vals[m].append(rec["metrics"][m])
            means = {m: float(np.mean(vals[m])) for m in METRICS}
            stds = {m: float(np.std(vals[m])) for m in METRICS}
            row = {"split": split, "variant": variant, "n_seeds": len(seed_runs)}
            for m in METRICS:
                row[f"{m}_mean"] = means[m]
                row[f"{m}_std"] = stds[m]
            rows.append(row)
            label = f"{split}_{variant}"
            print(f"{label:30s} " + " ".join(f"{means[m]:.3f}+/-{stds[m]:.3f}" for m in METRICS))

    fieldnames = ["split", "variant", "n_seeds"] + [f"{m}_{suf}" for m in METRICS for suf in ("mean", "std")]
    with open("results/summary_table.csv", "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
    print("Saved results/summary_table.csv")
    return rows


def plot_comparison(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(SPLITS))
    width = 0.2
    colors = {"baseline": "#8A93A6", "fingerprint": "#2F8F86",
              "random_fingerprint": "#D99A3E", "adaptive": "#C0392B"}

    for ax, metric in zip(axes, ["auroc", "auprc"]):
        for j, variant in enumerate(VARIANTS):
            means, stds = [], []
            for split in SPLITS:
                row = next(r for r in rows if r["split"] == split and r["variant"] == variant)
                means.append(row[f"{metric}_mean"])
                stds.append(row[f"{metric}_std"])
            ax.bar(x + (j - 1.5) * width, means, width, yerr=stds, capsize=3,
                   label=variant, color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(["random", "cold_start"])
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} (mean +/- SD, 3 seeds)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = "figures/comparison_auroc_auprc.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved {out_path}")


def evidence_stratification(records):
    """For the cold_start split, bucket each test edge's compound into
    Low/Medium/High non-CtD graph-degree tertiles (computed from that run's
    OWN per_edge 'degree' field -- already leakage-safe, train-graph-only),
    and report mean Hits@10 and mean rank per bucket per variant, averaged
    across the 3 seeds."""
    bucket_stats = defaultdict(lambda: defaultdict(list))  # (variant,bucket) -> metric -> [values across seeds]

    for seed in SEEDS:
        for variant in VARIANTS:
            rec = records[("cold_start", variant)].get(seed)
            if rec is None:
                continue
            per_edge = rec["per_edge"]
            degrees = np.array([e["degree"] for e in per_edge])
            # tertile cut points from THIS run's own test-compound degree distribution
            q1, q2 = np.percentile(degrees, [33.33, 66.67])

            per_bucket_hits10 = defaultdict(list)
            per_bucket_rank = defaultdict(list)
            for e in per_edge:
                d = e["degree"]
                bucket = "low" if d <= q1 else ("medium" if d <= q2 else "high")
                per_bucket_hits10[bucket].append(1.0 if e["rank"] <= 10 else 0.0)
                per_bucket_rank[bucket].append(e["rank"])

            for bucket in ["low", "medium", "high"]:
                if per_bucket_hits10[bucket]:
                    bucket_stats[(variant, bucket)]["hits@10"].append(np.mean(per_bucket_hits10[bucket]))
                    bucket_stats[(variant, bucket)]["mean_rank"].append(np.mean(per_bucket_rank[bucket]))
                    bucket_stats[(variant, bucket)]["n"].append(len(per_bucket_hits10[bucket]))

    rows = []
    print("\nEvidence stratification (cold_start split, mean +/- SD over 3 seeds):")
    print(f"{'variant':20s} {'bucket':8s} {'hits@10':>14s} {'mean_rank':>16s} {'avg_n':>8s}")
    for variant in VARIANTS:
        for bucket in ["low", "medium", "high"]:
            stats = bucket_stats[(variant, bucket)]
            if not stats:
                continue
            h10_mean, h10_std = np.mean(stats["hits@10"]), np.std(stats["hits@10"])
            rank_mean, rank_std = np.mean(stats["mean_rank"]), np.std(stats["mean_rank"])
            n_avg = np.mean(stats["n"])
            print(f"{variant:20s} {bucket:8s} {h10_mean:.3f}+/-{h10_std:.3f}   "
                  f"{rank_mean:7.2f}+/-{rank_std:5.2f}   {n_avg:6.1f}")
            rows.append({"variant": variant, "bucket": bucket,
                         "hits@10_mean": h10_mean, "hits@10_std": h10_std,
                         "mean_rank_mean": rank_mean, "mean_rank_std": rank_std,
                         "avg_n": n_avg})

    with open("results/evidence_stratification.csv", "w") as f:
        f.write("variant,bucket,hits@10_mean,hits@10_std,mean_rank_mean,mean_rank_std,avg_n\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in
                     ["variant", "bucket", "hits@10_mean", "hits@10_std",
                      "mean_rank_mean", "mean_rank_std", "avg_n"]) + "\n")
    print("Saved results/evidence_stratification.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"baseline": "#8A93A6", "fingerprint": "#2F8F86",
              "random_fingerprint": "#D99A3E", "adaptive": "#C0392B"}
    buckets = ["low", "medium", "high"]
    x = np.arange(len(buckets))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for j, variant in enumerate(VARIANTS):
        means = [next((r["hits@10_mean"] for r in rows if r["variant"] == variant and r["bucket"] == b), 0) for b in buckets]
        stds = [next((r["hits@10_std"] for r in rows if r["variant"] == variant and r["bucket"] == b), 0) for b in buckets]
        ax.bar(x + (j - 1.5) * width, means, width, yerr=stds, capsize=3, label=variant, color=colors[variant])
    ax.set_xticks(x)
    ax.set_xticklabels(["Low evidence", "Medium evidence", "High evidence"])
    ax.set_ylabel("Hits@10")
    ax.set_title("Cold-start Hits@10 by compound graph-evidence tertile (mean +/- SD, 3 seeds)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/evidence_stratification.png", dpi=130)
    plt.close()
    print("Saved figures/evidence_stratification.png")


def adaptive_ablation(records):
    """Novelty-control ablation: adaptive (real fingerprints) vs
    adaptive_random_fingerprint (same gated architecture, shuffled
    fingerprints), mean +/- SD over 3 seeds, both splits. Answers directly
    whether adaptive's gain is the gating mechanism alone or depends on the
    real compound<->structure correspondence."""
    rows = []
    print("\nAdaptive novelty-control ablation (mean +/- SD over 3 seeds):")
    print(f"{'run':40s} " + " ".join(f"{m:>14s}" for m in METRICS))
    for split in SPLITS:
        for variant in ABLATION_VARIANTS:
            seed_runs = records[(split, variant)]
            if not seed_runs:
                print(f"WARNING: no runs for ({split}, {variant})")
                continue
            vals = {m: [] for m in METRICS}
            for seed, rec in seed_runs.items():
                for m in METRICS:
                    vals[m].append(rec["metrics"][m])
            means = {m: float(np.mean(vals[m])) for m in METRICS}
            stds = {m: float(np.std(vals[m])) for m in METRICS}
            row = {"split": split, "variant": variant, "n_seeds": len(seed_runs)}
            for m in METRICS:
                row[f"{m}_mean"] = means[m]
                row[f"{m}_std"] = stds[m]
            rows.append(row)
            label = f"{split}_{variant}"
            print(f"{label:40s} " + " ".join(f"{means[m]:.3f}+/-{stds[m]:.3f}" for m in METRICS))

    fieldnames = ["split", "variant", "n_seeds"] + [f"{m}_{suf}" for m in METRICS for suf in ("mean", "std")]
    with open("results/adaptive_ablation_table.csv", "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fieldnames) + "\n")
    print("Saved results/adaptive_ablation_table.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    x = np.arange(len(SPLITS))
    width = 0.32
    colors = {"adaptive": "#C0392B", "adaptive_random_fingerprint": "#8A93A6"}
    labels = {"adaptive": "adaptive (real fp)", "adaptive_random_fingerprint": "adaptive (shuffled fp)"}

    for ax, metric in zip(axes, ["auroc", "auprc", "hits@10"]):
        for j, variant in enumerate(ABLATION_VARIANTS):
            means, stds = [], []
            for split in SPLITS:
                row = next((r for r in rows if r["split"] == split and r["variant"] == variant), None)
                means.append(row[f"{metric}_mean"] if row else 0)
                stds.append(row[f"{metric}_std"] if row else 0)
            ax.bar(x + (j - 0.5) * width, means, width, yerr=stds, capsize=3,
                   label=labels[variant], color=colors[variant])
        ax.set_xticks(x)
        ax.set_xticklabels(["random", "cold_start"])
        ax.set_ylabel(metric.upper() if metric != "hits@10" else "Hits@10")
        ax.set_title(metric.upper() if metric != "hits@10" else "Hits@10")
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out_path = "figures/adaptive_ablation.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved {out_path}")
    return rows


def main():
    records = load_all()
    rows = mean_std_table(records)
    plot_comparison(rows)
    evidence_stratification(records)
    adaptive_ablation(records)


if __name__ == "__main__":
    main()
