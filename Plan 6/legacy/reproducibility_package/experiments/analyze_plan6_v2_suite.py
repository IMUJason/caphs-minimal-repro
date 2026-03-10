from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_BASELINES = [
    "fcfs",
    "least_loaded",
    "fixed_priority",
    "work_stealing",
    "heft",
    "feedback_threshold",
]

POLICY_LABELS = {
    "plan6_v2": "CAPHS",
    "fcfs": "FCFS",
    "least_loaded": "Least loaded",
    "fixed_priority": "Fixed priority",
    "work_stealing": "Work stealing",
    "heft": "HEFT",
    "feedback_threshold": "Feedback threshold",
}


def exact_sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    threshold = max(wins, losses)
    tail = sum(math.comb(total, k) for k in range(threshold, total + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _pairwise_focus_vs_baseline(per_run: pd.DataFrame, focus_policy: str, baseline: str) -> dict[str, float | int | str]:
    focus = per_run.loc[per_run["policy"] == focus_policy].copy()
    other = per_run.loc[per_run["policy"] == baseline].copy()
    merged = focus.merge(
        other,
        on=["workload_name", "workload_seed"],
        suffixes=("_focus", "_baseline"),
    )
    if merged.empty:
        raise ValueError(f"No paired runs for {focus_policy} vs {baseline}")
    throughput_delta = merged["throughput_focus"] - merged["throughput_baseline"]
    p95_delta = merged["p95_latency_baseline"] - merged["p95_latency_focus"]
    energy_delta = merged["total_energy_baseline"] - merged["total_energy_focus"]
    focus_better_both = ((throughput_delta > 0) & (p95_delta > 0)).sum()
    throughput_wins = int((throughput_delta > 0).sum())
    throughput_losses = int((throughput_delta < 0).sum())
    p95_wins = int((p95_delta > 0).sum())
    p95_losses = int((p95_delta < 0).sum())
    energy_wins = int((energy_delta > 0).sum())
    energy_losses = int((energy_delta < 0).sum())
    return {
        "focus_policy": focus_policy,
        "baseline": baseline,
        "paired_runs": int(len(merged)),
        "throughput_delta_mean": round(float(throughput_delta.mean()), 6),
        "throughput_delta_pct": round(float(100.0 * throughput_delta.mean() / merged["throughput_baseline"].mean()), 4),
        "p95_latency_delta_mean": round(float(p95_delta.mean()), 4),
        "p95_latency_delta_pct": round(float(100.0 * p95_delta.mean() / merged["p95_latency_baseline"].mean()), 4),
        "energy_delta_mean": round(float(energy_delta.mean()), 4),
        "energy_delta_pct": round(float(100.0 * energy_delta.mean() / merged["total_energy_baseline"].mean()), 4),
        "joint_wins_throughput_and_p95": int(focus_better_both),
        "throughput_wins": throughput_wins,
        "throughput_losses": throughput_losses,
        "throughput_sign_pvalue": round(exact_sign_test(throughput_wins, throughput_losses), 6),
        "p95_wins": p95_wins,
        "p95_losses": p95_losses,
        "p95_sign_pvalue": round(exact_sign_test(p95_wins, p95_losses), 6),
        "energy_wins": energy_wins,
        "energy_losses": energy_losses,
        "energy_sign_pvalue": round(exact_sign_test(energy_wins, energy_losses), 6),
    }


def _tradeoff_figure(summary: pd.DataFrame, focus_policy: str, output_path: Path) -> None:
    highlight = {focus_policy, "fcfs", "least_loaded", "fixed_priority", "work_stealing", "heft", "feedback_threshold"}
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for _, row in summary.groupby("policy").agg(
        throughput_mean=("throughput_mean", "mean"),
        p95_latency_mean=("p95_latency_mean", "mean"),
    ).reset_index().iterrows():
        policy = row["policy"]
        color = "#d62728" if policy == focus_policy else ("#1f77b4" if policy in highlight else "#7f7f7f")
        size = 90 if policy == focus_policy else (70 if policy in highlight else 45)
        ax.scatter(row["throughput_mean"], row["p95_latency_mean"], s=size, color=color, alpha=0.85)
        ax.annotate(POLICY_LABELS.get(policy, policy), (row["throughput_mean"], row["p95_latency_mean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean throughput")
    ax.set_ylabel("Mean P95 latency")
    ax.set_title(f"{POLICY_LABELS.get(focus_policy, focus_policy)} throughput-latency trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _workload_figure(summary: pd.DataFrame, focus_policy: str, output_path: Path) -> None:
    selected_policies = [focus_policy, "fixed_priority", "work_stealing", "least_loaded", "fcfs"]
    selected = summary.loc[summary["policy"].isin(selected_policies)].copy()
    workloads = list(selected["workload_name"].drop_duplicates())
    policies = selected_policies
    colors = {
        focus_policy: "#d62728",
        "fcfs": "#1f77b4",
        "least_loaded": "#17a398",
        "fixed_priority": "#2ca02c",
        "work_stealing": "#9467bd",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    width = 0.15
    positions = list(range(len(workloads)))
    for offset, policy in enumerate(policies):
        subset = selected.loc[selected["policy"] == policy].set_index("workload_name")
        xs = [position + (offset - 2) * width for position in positions]
        axes[0].bar(xs, [subset.loc[workload, "throughput_mean"] for workload in workloads], width=width, color=colors[policy], label=POLICY_LABELS.get(policy, policy))
        axes[1].bar(xs, [subset.loc[workload, "p95_latency_mean"] for workload in workloads], width=width, color=colors[policy], label=POLICY_LABELS.get(policy, policy))
    for ax, metric in zip(axes, ["Mean throughput", "Mean P95 latency"]):
        ax.set_xticks(positions)
        ax.set_xticklabels(workloads, rotation=15)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Throughput by workload")
    axes[1].set_title("P95 latency by workload")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(policies), loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_suite(results_dir: str | Path, focus_policy: str = "plan6_v2", baselines: list[str] | None = None) -> dict[str, Path]:
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    per_run = pd.read_csv(results_dir / "per_run_metrics.csv")
    summary = pd.read_csv(results_dir / "summary_metrics.csv")
    baselines = baselines or [baseline for baseline in DEFAULT_BASELINES if baseline in set(per_run["policy"])]
    pairwise_rows = [_pairwise_focus_vs_baseline(per_run, focus_policy, baseline) for baseline in baselines]
    pairwise = pd.DataFrame(pairwise_rows).sort_values(["baseline"])
    pairwise_path = tables_dir / f"{focus_policy}_pairwise_vs_baselines.csv"
    pairwise.to_csv(pairwise_path, index=False)
    focus_summary = (
        summary.loc[summary["policy"] == focus_policy]
        .sort_values("workload_name")
        .reset_index(drop=True)
    )
    focus_summary_path = tables_dir / f"{focus_policy}_workload_summary.csv"
    focus_summary.to_csv(focus_summary_path, index=False)
    tradeoff_path = figures_dir / f"{focus_policy}_tradeoff.png"
    workload_path = figures_dir / f"{focus_policy}_workload_comparison.png"
    _tradeoff_figure(summary, focus_policy, tradeoff_path)
    _workload_figure(summary, focus_policy, workload_path)
    fcfs_comparison = pairwise.loc[pairwise["baseline"] == "fcfs"]
    ll_comparison = pairwise.loc[pairwise["baseline"] == "least_loaded"]
    fp_comparison = pairwise.loc[pairwise["baseline"] == "fixed_priority"]
    ws_comparison = pairwise.loc[pairwise["baseline"] == "work_stealing"]
    md_lines = [
        f"# {POLICY_LABELS.get(focus_policy, focus_policy)} Analysis Summary",
        "",
        f"Result directory: `{results_dir}`",
        "",
    ]
    if not fcfs_comparison.empty:
        row = fcfs_comparison.iloc[0]
        md_lines.extend(
            [
                f"{POLICY_LABELS.get(focus_policy, focus_policy)} improves mean throughput over `fcfs` by {row['throughput_delta_pct']:.4f}% "
                f"and reduces mean P95 latency by {row['p95_latency_delta_pct']:.4f}% across {int(row['paired_runs'])} paired runs.",
                "",
            ]
        )
    if not ll_comparison.empty:
        row = ll_comparison.iloc[0]
        md_lines.append(
            f"Against `least_loaded`, {POLICY_LABELS.get(focus_policy, focus_policy)} changes mean throughput by {row['throughput_delta_pct']:.4f}% "
            f"and improves mean P95 latency by {row['p95_latency_delta_pct']:.4f}%."
        )
        md_lines.append("")
    if not fp_comparison.empty:
        row = fp_comparison.iloc[0]
        md_lines.append(
            f"Against `fixed_priority`, {POLICY_LABELS.get(focus_policy, focus_policy)} changes mean throughput by {row['throughput_delta_pct']:.4f}% "
            f"and improves mean P95 latency by {row['p95_latency_delta_pct']:.4f}%."
        )
        md_lines.append("")
    if not ws_comparison.empty:
        row = ws_comparison.iloc[0]
        md_lines.append(
            f"Against `work_stealing`, {POLICY_LABELS.get(focus_policy, focus_policy)} changes mean throughput by {row['throughput_delta_pct']:.4f}% "
            f"and improves mean P95 latency by {row['p95_latency_delta_pct']:.4f}%."
        )
        md_lines.append("")
    md_lines.append("The paired sign tests are reported in the CSV tables for transparent audit and later manuscript use.")
    summary_path = results_dir / f"{focus_policy}_analysis_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    figure_manifest = pd.DataFrame(
        [
            {
                "figure_id": f"{focus_policy}_tradeoff",
                "path": str(tradeoff_path),
                "description": f"Throughput-latency trade-off scatter for {focus_policy} against all policies",
                "source_csv": str(results_dir / "summary_metrics.csv"),
                "status": "generated",
            },
            {
                "figure_id": f"{focus_policy}_workload_comparison",
                "path": str(workload_path),
                "description": f"Per-workload throughput and P95 comparison for {focus_policy} and strongest baselines",
                "source_csv": str(results_dir / "summary_metrics.csv"),
                "status": "generated",
            },
        ]
    )
    figure_manifest_path = results_dir / "figure_manifest.csv"
    figure_manifest.to_csv(figure_manifest_path, index=False)
    table_manifest = pd.DataFrame(
        [
            {
                "table_id": f"{focus_policy}_pairwise_vs_baselines",
                "path": str(pairwise_path),
                "description": f"Paired comparison table for {focus_policy} against baselines",
                "source_csv": str(results_dir / "per_run_metrics.csv"),
                "status": "generated",
            },
            {
                "table_id": f"{focus_policy}_workload_summary",
                "path": str(focus_summary_path),
                "description": f"Per-workload aggregate summary for {focus_policy}",
                "source_csv": str(results_dir / "summary_metrics.csv"),
                "status": "generated",
            },
        ]
    )
    table_manifest_path = results_dir / "table_manifest.csv"
    table_manifest.to_csv(table_manifest_path, index=False)
    return {
        "pairwise_table": pairwise_path,
        "focus_summary_table": focus_summary_path,
        "tradeoff_figure": tradeoff_path,
        "workload_figure": workload_path,
        "markdown_summary": summary_path,
        "figure_manifest": figure_manifest_path,
        "table_manifest": table_manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--focus-policy", default="plan6_v2")
    parser.add_argument("--baselines", nargs="*", default=None)
    args = parser.parse_args()
    outputs = analyze_suite(args.results_dir, focus_policy=args.focus_policy, baselines=args.baselines)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
