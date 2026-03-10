from __future__ import annotations

from pathlib import Path

from plan6_scheduler.simulator import simulate_dataset
from plan6_scheduler.workloads import create_workload_file


def test_simulation_runs_and_completes(tmp_path: Path) -> None:
    workload_path = tmp_path / "steady.json"
    create_workload_file(workload_path, "steady_mixed", seed=7)
    result = simulate_dataset(workload_path, policy_name="plan6", seed=17)
    assert result.summary["task_count"] > 0
    assert result.summary["makespan"] > 0
    assert result.summary["throughput"] > 0
    assert len(result.timeline) > 0


def test_heft_and_fcfs_differ(tmp_path: Path) -> None:
    workload_path = tmp_path / "dag.json"
    create_workload_file(workload_path, "dag_stream", seed=3)
    heft_result = simulate_dataset(workload_path, policy_name="heft", seed=11)
    fcfs_result = simulate_dataset(workload_path, policy_name="fcfs", seed=11)
    assert heft_result.summary["makespan"] != fcfs_result.summary["makespan"]


def test_plan6_v2_improves_dag_tail_on_reference_seed(tmp_path: Path) -> None:
    workload_path = tmp_path / "dag_v2.json"
    create_workload_file(workload_path, "dag_stream", seed=1)
    plan6_result = simulate_dataset(workload_path, policy_name="plan6", seed=123456)
    plan6_v2_result = simulate_dataset(workload_path, policy_name="plan6_v2", seed=123456)
    assert plan6_v2_result.summary["throughput"] >= plan6_result.summary["throughput"]
    assert plan6_v2_result.summary["p95_latency"] <= plan6_result.summary["p95_latency"]
