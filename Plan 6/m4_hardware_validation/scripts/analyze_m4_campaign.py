from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw_runs"
MONITOR_DIR = RESULTS_DIR / "monitoring"


def percentile(values: pd.Series, q: float) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(q))


def pairwise_vs_baseline(summary: pd.DataFrame, focus: str, baseline: str) -> dict:
    focus_frame = summary[summary["policy"] == focus].set_index("workload_id")
    base_frame = summary[summary["policy"] == baseline].set_index("workload_id")
    paired = focus_frame.join(base_frame, lsuffix="_focus", rsuffix="_base", how="inner")
    throughput_delta_pct = ((paired["throughput_tasks_per_s_focus"] - paired["throughput_tasks_per_s_base"]) / paired["throughput_tasks_per_s_base"] * 100.0)
    p95_delta_pct = ((paired["p95_latency_ms_base"] - paired["p95_latency_ms_focus"]) / paired["p95_latency_ms_base"] * 100.0)
    power_delta_pct = ((paired["mean_power_impact_base"] - paired["mean_power_impact_focus"]) / paired["mean_power_impact_base"] * 100.0)
    return {
        "baseline": baseline,
        "paired_runs": int(len(paired)),
        "throughput_delta_pct": float(throughput_delta_pct.mean()),
        "p95_latency_delta_pct": float(p95_delta_pct.mean()),
        "power_impact_delta_pct": float(power_delta_pct.mean()),
        "joint_wins": int(((throughput_delta_pct > 0) & (p95_delta_pct > 0)).sum()),
        "throughput_wins": int((throughput_delta_pct > 0).sum()),
        "throughput_losses": int((throughput_delta_pct < 0).sum()),
        "p95_wins": int((p95_delta_pct > 0).sum()),
        "p95_losses": int((p95_delta_pct < 0).sum()),
        "power_wins": int((power_delta_pct > 0).sum()),
        "power_losses": int((power_delta_pct < 0).sum()),
    }


def main() -> None:
    raw_rows = []
    task_rows = []
    chunk_rows = []
    for result_path in sorted(RAW_DIR.glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        run_id = result_path.stem
        monitor_path = MONITOR_DIR / f"{run_id}_monitor.json"
        monitor_payload = json.loads(monitor_path.read_text(encoding="utf-8"))
        summary["run_id"] = run_id
        summary["mean_power_impact"] = monitor_payload["summary"]["mean_power_impact"]
        summary["peak_power_impact"] = monitor_payload["summary"]["peak_power_impact"]
        summary["mean_cpu_pct"] = monitor_payload["summary"]["mean_cpu_pct"]
        raw_rows.append(summary)
        for task in payload["task_metrics"]:
            task["run_id"] = run_id
            task["policy"] = summary["policy"]
            task["workload_id"] = summary["workload_id"]
            task_rows.append(task)
        for chunk in payload["chunk_events"]:
            chunk["run_id"] = run_id
            chunk["policy"] = summary["policy"]
            chunk["workload_id"] = summary["workload_id"]
            chunk_rows.append(chunk)

    summary_df = pd.DataFrame(raw_rows)
    task_df = pd.DataFrame(task_rows)
    chunk_df = pd.DataFrame(chunk_rows)
    summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    task_df.to_csv(RESULTS_DIR / "task_metrics.csv", index=False)
    chunk_df.to_csv(RESULTS_DIR / "chunk_events.csv", index=False)

    aggregate = summary_df.groupby("policy", as_index=False).agg(
        throughput_tasks_per_s=("throughput_tasks_per_s", "mean"),
        mean_latency_ms=("mean_latency_ms", "mean"),
        p95_latency_ms=("p95_latency_ms", "mean"),
        mean_power_impact=("mean_power_impact", "mean"),
        scheduler_mean_us=("scheduler_mean_us", "mean"),
        migrations_total=("migrations_total", "mean"),
        gpu_chunks=("gpu_chunks", "mean"),
        cpu_chunks=("cpu_chunks", "mean"),
    )
    aggregate.to_csv(RESULTS_DIR / "aggregate_policy_metrics.csv", index=False)

    baselines = ["fcfs", "least_loaded", "fixed_priority", "work_stealing"]
    pairwise_rows = [pairwise_vs_baseline(summary_df, "caphs", baseline) for baseline in baselines]
    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(RESULTS_DIR / "caphs_pairwise_vs_baselines.csv", index=False)

    by_family = summary_df.groupby(["family", "policy"], as_index=False).agg(
        throughput_tasks_per_s=("throughput_tasks_per_s", "mean"),
        p95_latency_ms=("p95_latency_ms", "mean"),
        mean_power_impact=("mean_power_impact", "mean"),
    )
    by_family.to_csv(RESULTS_DIR / "family_policy_metrics.csv", index=False)

    lines = []
    lines.append("# Apple M4 Pro Hardware Validation Summary\n")
    lines.append(f"- Total runs: `{len(summary_df)}`")
    lines.append(f"- Workload instances: `{summary_df['workload_id'].nunique()}`")
    lines.append(f"- Policies: `{', '.join(sorted(summary_df['policy'].unique()))}`\n")
    lines.append("## Aggregate metrics\n")
    lines.append(aggregate.to_markdown(index=False))
    lines.append("\n## CAPHS pairwise improvements\n")
    lines.append(pairwise_df.to_markdown(index=False))
    (RESULTS_DIR / "hardware_validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'hardware_validation_summary.md'}")


if __name__ == "__main__":
    main()
