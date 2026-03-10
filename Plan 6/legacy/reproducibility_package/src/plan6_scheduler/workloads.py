from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import RESOURCE_TYPES


TASK_LIBRARY: dict[str, dict[str, Any]] = {
    "cpu_control": {
        "latency_class": "latency_sensitive",
        "priority_range": (7, 10),
        "size_range": (22, 44),
        "bandwidth_range": (1.0, 2.4),
        "checkpoint_interval": 0.25,
        "migration_penalty": 2,
        "service_scale": {"cpu": 1.0, "gpu": 1.9, "npu": 2.5},
        "power_scale": {"cpu": 1.0, "gpu": 1.35, "npu": 1.15},
    },
    "gpu_dense": {
        "latency_class": "throughput_oriented",
        "priority_range": (3, 8),
        "size_range": (48, 92),
        "bandwidth_range": (2.0, 4.6),
        "checkpoint_interval": 0.2,
        "migration_penalty": 3,
        "service_scale": {"cpu": 2.6, "gpu": 0.85, "npu": 1.45},
        "power_scale": {"cpu": 0.95, "gpu": 1.55, "npu": 1.2},
    },
    "npu_infer": {
        "latency_class": "interactive",
        "priority_range": (6, 10),
        "size_range": (16, 32),
        "bandwidth_range": (0.8, 2.0),
        "checkpoint_interval": 0.2,
        "migration_penalty": 2,
        "service_scale": {"cpu": 1.8, "gpu": 1.25, "npu": 0.7},
        "power_scale": {"cpu": 0.75, "gpu": 1.05, "npu": 0.8},
    },
    "memory_bound": {
        "latency_class": "best_effort",
        "priority_range": (2, 6),
        "size_range": (28, 58),
        "bandwidth_range": (3.5, 5.2),
        "checkpoint_interval": 0.33,
        "migration_penalty": 4,
        "service_scale": {"cpu": 1.2, "gpu": 1.35, "npu": 1.55},
        "power_scale": {"cpu": 0.9, "gpu": 1.1, "npu": 0.95},
    },
}


WORKLOAD_PROFILES: dict[str, dict[str, Any]] = {
    "steady_mixed": {
        "task_count": 120,
        "arrival_rate": 1.2,
        "mix": {"cpu_control": 0.34, "gpu_dense": 0.26, "npu_infer": 0.22, "memory_bound": 0.18},
        "dag_probability": 0.35,
        "burst_pattern": [],
    },
    "bursty_mixed": {
        "task_count": 140,
        "arrival_rate": 1.0,
        "mix": {"cpu_control": 0.28, "gpu_dense": 0.28, "npu_infer": 0.18, "memory_bound": 0.26},
        "dag_probability": 0.40,
        "burst_pattern": [(20, 34, 3.4), (74, 92, 2.6)],
    },
    "thermal_pressure": {
        "task_count": 135,
        "arrival_rate": 1.15,
        "mix": {"cpu_control": 0.18, "gpu_dense": 0.42, "npu_infer": 0.24, "memory_bound": 0.16},
        "dag_probability": 0.30,
        "burst_pattern": [(40, 70, 2.3)],
    },
    "dag_stream": {
        "task_count": 128,
        "arrival_rate": 1.05,
        "mix": {"cpu_control": 0.25, "gpu_dense": 0.24, "npu_infer": 0.20, "memory_bound": 0.31},
        "dag_probability": 0.72,
        "burst_pattern": [(50, 75, 1.9)],
    },
}


def _weighted_choice(rng: np.random.Generator, mix: dict[str, float]) -> str:
    keys = list(mix)
    probs = np.array([mix[key] for key in keys], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(keys, p=probs))


def _arrival_sequence(profile: dict[str, Any], task_count: int, rng: np.random.Generator) -> list[int]:
    current_time = 0
    arrivals: list[int] = []
    rate = profile["arrival_rate"]
    bursts = profile["burst_pattern"]
    for _ in range(task_count):
        multiplier = 1.0
        for start, end, value in bursts:
            if start <= current_time <= end:
                multiplier = value
                break
        gap = max(1, int(rng.poisson(rate / multiplier) + 1))
        current_time += gap
        arrivals.append(current_time)
    return arrivals


def _service_time(size: float, task_profile: dict[str, Any], rng: np.random.Generator) -> dict[str, int]:
    base = size * rng.uniform(0.9, 1.1)
    times: dict[str, int] = {}
    for resource_type in RESOURCE_TYPES:
        duration = int(np.ceil(base * task_profile["service_scale"][resource_type] / 5.5))
        times[resource_type] = max(1, duration)
    return times


def _power_draw(size: float, bandwidth: float, task_profile: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    baseline = 4.0 + size / 20.0 + bandwidth * 0.7 + rng.uniform(-0.3, 0.3)
    return {
        resource_type: round(max(1.0, baseline * task_profile["power_scale"][resource_type]), 3)
        for resource_type in RESOURCE_TYPES
    }


def _generate_dag_blocks(
    task_ids: list[str],
    profile: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    deps = {task_id: [] for task_id in task_ids}
    if len(task_ids) < 3:
        return deps
    dag_probability = profile["dag_probability"]
    index = 0
    while index < len(task_ids):
        if rng.random() > dag_probability:
            index += 1
            continue
        block_size = int(rng.integers(3, 7))
        block = task_ids[index : index + block_size]
        if len(block) < 3:
            break
        for local_index in range(1, len(block)):
            parent = block[local_index - 1]
            child = block[local_index]
            deps[child].append(parent)
            if local_index >= 2 and rng.random() < 0.35:
                deps[child].append(block[local_index - 2])
        index += block_size
    return deps


def generate_workload(workload_name: str, seed: int) -> dict[str, Any]:
    if workload_name not in WORKLOAD_PROFILES:
        raise KeyError(f"Unknown workload {workload_name}")
    rng = np.random.default_rng(seed)
    profile = WORKLOAD_PROFILES[workload_name]
    task_count = profile["task_count"]
    arrivals = _arrival_sequence(profile, task_count, rng)
    task_ids = [f"{workload_name}-task-{idx:03d}" for idx in range(task_count)]
    deps_map = _generate_dag_blocks(task_ids, profile, rng)
    tasks: list[dict[str, Any]] = []
    current_dag = 0
    for task_id, arrival in zip(task_ids, arrivals):
        task_kind = _weighted_choice(rng, profile["mix"])
        task_profile = TASK_LIBRARY[task_kind]
        size = float(rng.uniform(*task_profile["size_range"]))
        bandwidth = float(rng.uniform(*task_profile["bandwidth_range"]))
        priority = int(rng.integers(task_profile["priority_range"][0], task_profile["priority_range"][1] + 1))
        if deps_map[task_id]:
            dag_id = f"{workload_name}-dag-{current_dag:03d}"
        else:
            current_dag += 1
            dag_id = f"{workload_name}-dag-{current_dag:03d}"
        tasks.append(
            {
                "task_id": task_id,
                "dag_id": dag_id,
                "arrival_time": int(arrival),
                "priority": priority,
                "latency_class": task_profile["latency_class"],
                "task_type": task_kind,
                "size": round(size, 3),
                "bandwidth_demand": round(bandwidth, 3),
                "checkpoint_interval": task_profile["checkpoint_interval"],
                "migration_penalty": task_profile["migration_penalty"],
                "service_time": _service_time(size, task_profile, rng),
                "power_draw": _power_draw(size, bandwidth, task_profile, rng),
                "deps": deps_map[task_id],
            }
        )
    return {
        "metadata": {
            "workload_name": workload_name,
            "seed": seed,
            "task_count": task_count,
            "profile": profile,
        },
        "tasks": tasks,
    }


def create_workload_file(output_path: str | Path, workload_name: str, seed: int) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_workload(workload_name, seed)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
