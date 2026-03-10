from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_all_figures(results_dir: str | Path) -> dict[str, Path]:
    results_root = Path(results_dir)
    per_run = pd.read_csv(results_root / "per_run_metrics.csv")
    summary = pd.read_csv(results_root / "summary_metrics.csv")
    figures_dir = results_root / "figures"
    outputs: dict[str, Path] = {}

    melted = per_run.melt(
        id_vars=["policy", "workload_name"],
        value_vars=["throughput", "p95_latency", "total_energy", "peak_temperature"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=melted, x="metric", y="value", hue="policy", ax=ax)
    ax.set_title("Overall performance comparison across workloads")
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    outputs["Figure 1"] = figures_dir / "figure1_overall_performance.png"
    _save(fig, outputs["Figure 1"])

    burst_frame = []
    for policy in ["plan6", "feedback_threshold", "least_loaded", "heft"]:
        timeline_path = next((results_root / "raw_runs").glob(f"R-{policy}-D-bursty_mixed-s01/timeline.csv"))
        frame = pd.read_csv(timeline_path)
        frame["policy"] = policy
        burst_frame.append(frame)
    burst_df = pd.concat(burst_frame, ignore_index=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    sns.lineplot(data=burst_df, x="time", y="ready_tasks", hue="policy", ax=axes[0])
    axes[0].set_ylabel("Ready tasks")
    sns.lineplot(data=burst_df, x="time", y="gpu_temp", hue="policy", ax=axes[1], legend=False)
    axes[1].set_ylabel("GPU temp")
    sns.lineplot(data=burst_df, x="time", y="total_power", hue="policy", ax=axes[2], legend=False)
    axes[2].set_ylabel("Power")
    axes[2].set_xlabel("Simulation time")
    outputs["Figure 2"] = figures_dir / "figure2_burst_response.png"
    _save(fig, outputs["Figure 2"])

    migration_df = per_run[per_run["policy"].isin(["plan6", "plan6_deterministic", "plan6_nomigration", "feedback_threshold"])]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=migration_df,
        x="migration_overhead_time",
        y="migration_benefit_ratio",
        hue="policy",
        style="workload_name",
        s=120,
        ax=ax,
    )
    ax.set_title("Migration overhead versus migration benefit")
    outputs["Figure 3"] = figures_dir / "figure3_migration_tradeoff.png"
    _save(fig, outputs["Figure 3"])

    energy_df = per_run[per_run["policy"].isin(["plan6", "feedback_threshold", "least_loaded", "heft", "work_stealing"])]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=energy_df,
        x="total_energy",
        y="throughput",
        hue="policy",
        style="workload_name",
        s=120,
        ax=ax,
    )
    ax.set_title("Energy-delay style comparison")
    outputs["Figure 4"] = figures_dir / "figure4_energy_delay.png"
    _save(fig, outputs["Figure 4"])

    ablation = per_run[
        per_run["policy"].isin(["plan6", "plan6_nomigration", "plan6_nofeedback", "plan6_nothermal", "plan6_noenergy", "plan6_deterministic"])
    ]
    ablation_melted = ablation.melt(
        id_vars=["policy", "workload_name"],
        value_vars=["throughput", "p95_latency", "peak_temperature"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=ablation_melted, x="metric", y="value", hue="policy", ax=ax)
    ax.set_title("Ablation study for Plan 6 variants")
    outputs["Figure 5"] = figures_dir / "figure5_ablation.png"
    _save(fig, outputs["Figure 5"])

    heatmap = summary.pivot(index="policy", columns="workload_name", values="peak_temperature_mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(heatmap, annot=True, fmt=".1f", cmap="magma", ax=ax)
    ax.set_title("Peak temperature sensitivity across workloads")
    outputs["Figure 6"] = figures_dir / "figure6_temperature_heatmap.png"
    _save(fig, outputs["Figure 6"])

    return outputs
