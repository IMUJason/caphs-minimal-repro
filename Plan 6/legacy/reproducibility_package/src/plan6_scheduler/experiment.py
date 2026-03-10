from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .simulator import simulate_dataset
from .workloads import WORKLOAD_PROFILES, create_workload_file


@dataclass
class ExperimentConfig:
    package_root: Path
    project_root: Path
    data_dir: Path
    results_dir: Path
    logs_dir: Path
    workload_names: list[str]
    workload_seeds: list[int]
    policies: list[str]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ExperimentConfig":
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        package_root = Path(config_path).resolve().parents[2]
        project_root = package_root.parents[2]
        return cls(
            package_root=package_root,
            project_root=project_root,
            data_dir=(project_root / payload["paths"]["data_dir"]).resolve(),
            results_dir=(project_root / payload["paths"]["results_dir"]).resolve(),
            logs_dir=(project_root / payload["paths"]["logs_dir"]).resolve(),
            workload_names=list(payload["workloads"]["names"]),
            workload_seeds=list(payload["workloads"]["seeds"]),
            policies=list(payload["policies"]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _ensure_dirs(config: ExperimentConfig) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "raw_runs").mkdir(parents=True, exist_ok=True)
    (config.results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (config.results_dir / "tables").mkdir(parents=True, exist_ok=True)


def _generate_workloads(config: ExperimentConfig) -> list[dict[str, Any]]:
    dataset_rows: list[dict[str, Any]] = []
    for workload_name in config.workload_names:
        if workload_name not in WORKLOAD_PROFILES:
            raise KeyError(f"Unknown workload {workload_name}")
        for seed in config.workload_seeds:
            dataset_id = f"D-{workload_name}-s{seed:02d}"
            output_path = config.data_dir / "workloads" / f"{dataset_id}.json"
            create_workload_file(output_path, workload_name, seed)
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_type": "synthetic_trace",
                    "source_path_or_url": str(output_path.relative_to(config.project_root)),
                    "collection_or_generation_date": pd.Timestamp.now().isoformat(),
                    "collector_or_generator": "plan6_scheduler.workloads.generate_workload",
                    "license_or_usage_note": "Generated in-project for reproducible simulation",
                    "raw_checksum": _sha256(output_path),
                    "preprocess_script": "none",
                    "preprocess_output_path": str(output_path.relative_to(config.project_root)),
                    "split_or_sampling_rule": f"workload={workload_name}; seed={seed}",
                    "status": "generated",
                    "notes": "Round 2 synthetic workload",
                }
            )
    return dataset_rows


def run_experiment_suite(config_path: str | Path) -> dict[str, Path]:
    config = ExperimentConfig.from_yaml(config_path)
    _ensure_dirs(config)
    dataset_rows = _generate_workloads(config)
    dataset_frame = pd.DataFrame(dataset_rows).sort_values("dataset_id")
    dataset_frame.to_csv(config.results_dir / "dataset_register.csv", index=False)
    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset_row in dataset_rows:
        dataset_path = config.project_root / dataset_row["source_path_or_url"]
        for policy_name in config.policies:
            run_id = f"R-{policy_name}-{dataset_row['dataset_id']}"
            run_dir = config.results_dir / "raw_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result = simulate_dataset(dataset_path, policy_name=policy_name, seed=_stable_seed(run_id))
            summary_path = run_dir / "summary.json"
            timeline_path = run_dir / "timeline.csv"
            events_path = run_dir / "events.csv"
            summary_path.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
            result.timeline.to_csv(timeline_path, index=False)
            result.events.to_csv(events_path, index=False)
            summary_rows.append(result.summary | {"run_id": run_id, "dataset_id": dataset_row["dataset_id"]})
            run_rows.append(
                {
                    "run_id": run_id,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "git_commit_or_snapshot_id": os.environ.get("GIT_COMMIT", "workspace-uncommitted"),
                    "config_path": str(Path(config_path).resolve().relative_to(config.project_root)),
                    "dataset_ids": dataset_row["dataset_id"],
                    "baseline_name": policy_name,
                    "hardware_profile": "simulation-single-node-heterogeneous",
                    "software_environment": f"python-{os.sys.version.split()[0]}",
                    "random_seed": _stable_seed(run_id),
                    "stdout_log_path": str(events_path.relative_to(config.project_root)),
                    "raw_result_path": str(events_path.relative_to(config.project_root)),
                    "processed_result_path": str(summary_path.relative_to(config.project_root)),
                    "figure_outputs": "",
                    "owner": "codex",
                    "status": "completed",
                    "notes": dataset_row["dataset_id"],
                }
            )
    summary_frame = pd.DataFrame(summary_rows).sort_values(["workload_name", "policy", "workload_seed"])
    summary_frame.to_csv(config.results_dir / "per_run_metrics.csv", index=False)
    grouped = (
        summary_frame.groupby(["policy", "workload_name"], as_index=False)
        .agg(
            makespan_mean=("makespan", "mean"),
            throughput_mean=("throughput", "mean"),
            average_latency_mean=("average_latency", "mean"),
            p95_latency_mean=("p95_latency", "mean"),
            total_energy_mean=("total_energy", "mean"),
            peak_temperature_mean=("peak_temperature", "mean"),
            migrations_mean=("number_of_migrations", "mean"),
            migration_benefit_ratio_mean=("migration_benefit_ratio", "mean"),
        )
    )
    grouped.to_csv(config.results_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(run_rows).sort_values("run_id").to_csv(config.results_dir / "run_register.csv", index=False)
    return {
        "dataset_register": config.results_dir / "dataset_register.csv",
        "run_register": config.results_dir / "run_register.csv",
        "per_run_metrics": config.results_dir / "per_run_metrics.csv",
        "summary_metrics": config.results_dir / "summary_metrics.csv",
    }
