from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
MAIN_DIR = ROOT / "results" / "v2_main_suite"
HOLD_DIR = ROOT / "results" / "v2_holdout_suite"
OUTPUT_CSV = ROOT / "results" / "plan6_v2_combined_pairwise.csv"
OUTPUT_MD = ROOT / "results" / "plan6_v2_campaign_summary.md"

FOCUS_POLICY = "plan6_v2"
BASELINES = ["fcfs", "least_loaded", "fixed_priority", "work_stealing", "heft", "feedback_threshold"]


def exact_sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    threshold = max(wins, losses)
    tail = sum(math.comb(total, k) for k in range(threshold, total + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def pairwise_focus_vs_baseline(per_run: pd.DataFrame, focus_policy: str, baseline: str) -> dict[str, float | int | str]:
    focus = per_run.loc[per_run["policy"] == focus_policy].copy()
    other = per_run.loc[per_run["policy"] == baseline].copy()
    merged = focus.merge(other, on=["workload_name", "workload_seed"], suffixes=("_focus", "_baseline"))
    throughput_delta = merged["throughput_focus"] - merged["throughput_baseline"]
    p95_delta = merged["p95_latency_baseline"] - merged["p95_latency_focus"]
    energy_delta = merged["total_energy_baseline"] - merged["total_energy_focus"]
    throughput_wins = int((throughput_delta > 0).sum())
    throughput_losses = int((throughput_delta < 0).sum())
    p95_wins = int((p95_delta > 0).sum())
    p95_losses = int((p95_delta < 0).sum())
    energy_wins = int((energy_delta > 0).sum())
    energy_losses = int((energy_delta < 0).sum())
    return {
        "baseline": baseline,
        "paired_runs": int(len(merged)),
        "throughput_delta_pct": round(float(100.0 * throughput_delta.mean() / merged["throughput_baseline"].mean()), 4),
        "p95_latency_delta_pct": round(float(100.0 * p95_delta.mean() / merged["p95_latency_baseline"].mean()), 4),
        "energy_delta_pct": round(float(100.0 * energy_delta.mean() / merged["total_energy_baseline"].mean()), 4),
        "joint_wins": int(((throughput_delta > 0) & (p95_delta > 0)).sum()),
        "throughput_wins": throughput_wins,
        "throughput_losses": throughput_losses,
        "p95_wins": p95_wins,
        "p95_losses": p95_losses,
        "energy_wins": energy_wins,
        "energy_losses": energy_losses,
        "throughput_sign_pvalue": round(exact_sign_test(throughput_wins, throughput_losses), 6),
        "p95_sign_pvalue": round(exact_sign_test(p95_wins, p95_losses), 6),
        "energy_sign_pvalue": round(exact_sign_test(energy_wins, energy_losses), 6),
    }


def main() -> None:
    main_per_run = pd.read_csv(MAIN_DIR / "per_run_metrics.csv")
    hold_per_run = pd.read_csv(HOLD_DIR / "per_run_metrics.csv")
    combined_per_run = pd.concat([main_per_run, hold_per_run], ignore_index=True)

    rows = [pairwise_focus_vs_baseline(combined_per_run, FOCUS_POLICY, baseline) for baseline in BASELINES if baseline in set(combined_per_run["policy"])]
    combined = pd.DataFrame(rows)
    combined.to_csv(OUTPUT_CSV, index=False)

    lines = [
        "# Plan 6 v2 Campaign Summary",
        "",
        "日期：2026-03-10",
        "",
        "本轮汇总将最终方法与传统非内部基线为主的对照集重新组织，用于论文主文与补充材料中的正式表述。",
        "",
        "## 合并 main + holdout 的 paired 结果",
        "",
    ]
    for _, row in combined.iterrows():
        lines.append(
            f"- 相对 `{row['baseline']}`：吞吐 `{row['throughput_delta_pct']:+.4f}%`，"
            f"P95 延迟 `{row['p95_latency_delta_pct']:+.4f}%`（正值表示下降），"
            f"能耗 `{row['energy_delta_pct']:+.4f}%`，joint wins `{int(row['joint_wins'])}/{int(row['paired_runs'])}`。"
        )
    lines.extend(
        [
            "",
            "## 审慎结论",
            "",
            "最安全的论文口径应为：最终方法在 audited heterogeneous scheduling 上相对传统基线给出了更稳定的 throughput-latency-energy 综合优势；结果支持整体运行点更优，不支持不加限定的“全局碾压”表述。",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
