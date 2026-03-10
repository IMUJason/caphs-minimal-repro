from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_DIR = ROOT / "workloads"
RESULTS_DIR = ROOT / "results"


@dataclass
class FamilyConfig:
    name: str
    task_count: int
    size_weights: list[tuple[int, float]]
    chunk_range: tuple[int, int]
    arrival_mode: str
    dag_probability: float


FAMILIES = [
    FamilyConfig(
        name="steady_mixed",
        task_count=28,
        size_weights=[(768, 0.28), (1024, 0.30), (1536, 0.24), (2048, 0.18)],
        chunk_range=(90, 130),
        arrival_mode="steady",
        dag_probability=0.20,
    ),
    FamilyConfig(
        name="bursty_mixed",
        task_count=30,
        size_weights=[(768, 0.22), (1024, 0.24), (1536, 0.28), (2048, 0.26)],
        chunk_range=(80, 120),
        arrival_mode="bursty",
        dag_probability=0.18,
    ),
    FamilyConfig(
        name="unified_memory_pressure",
        task_count=24,
        size_weights=[(1024, 0.18), (1536, 0.34), (2048, 0.48)],
        chunk_range=(70, 110),
        arrival_mode="pressure",
        dag_probability=0.12,
    ),
    FamilyConfig(
        name="dag_stream",
        task_count=26,
        size_weights=[(768, 0.16), (1024, 0.24), (1536, 0.34), (2048, 0.26)],
        chunk_range=(60, 90),
        arrival_mode="dag",
        dag_probability=0.42,
    ),
]

SEEDS = [101, 202, 303]


def weighted_choice(rng: random.Random, items: list[tuple[int, float]]) -> int:
    x = rng.random()
    total = 0.0
    for value, weight in items:
        total += weight
        if x <= total:
            return value
    return items[-1][0]


def normalize_criticality(parents: dict[str, list[str]], tasks: list[str]) -> dict[str, float]:
    children: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    for task_id, task_parents in parents.items():
        for parent in task_parents:
            children[parent].append(task_id)

    order = tasks[:]
    depth_to_sink: dict[str, int] = {task_id: 0 for task_id in tasks}
    for task_id in reversed(order):
        if children[task_id]:
            depth_to_sink[task_id] = 1 + max(depth_to_sink[child] for child in children[task_id])
    max_depth = max(depth_to_sink.values()) if depth_to_sink else 1
    return {
        task_id: (depth_to_sink[task_id] / max_depth if max_depth > 0 else 0.0)
        for task_id in tasks
    }


def arrival_time_ms(rng: random.Random, family: FamilyConfig, index: int) -> float:
    if family.arrival_mode == "steady":
        return index * rng.uniform(6.5, 9.5)
    if family.arrival_mode == "bursty":
        if 8 <= index <= 14 or 20 <= index <= 26:
            return (index // 2) * rng.uniform(2.0, 4.0)
        return index * rng.uniform(7.0, 11.0)
    if family.arrival_mode == "pressure":
        return index * rng.uniform(4.0, 7.0)
    return index * rng.uniform(5.5, 8.5)


def build_workload(family: FamilyConfig, seed: int) -> dict:
    rng = random.Random(seed)
    specs = []
    task_ids = [f"T{index:03d}" for index in range(family.task_count)]
    parents: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    for idx, task_id in enumerate(task_ids):
        if idx > 1:
            for parent_idx in range(idx):
                if rng.random() < family.dag_probability * math.exp(-(idx - parent_idx) / 6.0):
                    parents[task_id].append(task_ids[parent_idx])
            parents[task_id] = sorted(set(parents[task_id]))[:3]
        size = weighted_choice(rng, family.size_weights)
        chunks = rng.randint(*family.chunk_range)
        priority = rng.randint(1, 5)
        arrival_ms_value = round(arrival_time_ms(rng, family, idx), 3)
        bytes_moved = float(3 * size * size * 4)
        specs.append(
            {
                "task_id": task_id,
                "size": size,
                "total_chunks": chunks,
                "priority": priority,
                "arrival_ms": arrival_ms_value,
                "parents": parents[task_id],
                "family": family.name,
                "bytes_moved": bytes_moved,
            }
        )
    criticality = normalize_criticality(parents, task_ids)
    for spec in specs:
        spec["criticality"] = round(criticality[spec["task_id"]], 6)

    return {
        "workload_id": f"{family.name}_s{seed}",
        "family": family.name,
        "seed": seed,
        "notes": "Real Apple M4 Pro hardware validation workload using chunked shared-memory GEMM tasks.",
        "task_specs": specs,
    }


def main() -> None:
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in FAMILIES:
        for seed in SEEDS:
            workload = build_workload(family, seed)
            path = WORKLOAD_DIR / f"{workload['workload_id']}.json"
            path.write_text(json.dumps(workload, indent=2), encoding="utf-8")
            rows.append(
                {
                    "workload_id": workload["workload_id"],
                    "family": family.name,
                    "seed": seed,
                    "task_count": len(workload["task_specs"]),
                    "output_path": str(path.relative_to(ROOT)),
                }
            )
    register_path = RESULTS_DIR / "workload_register.json"
    register_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} workloads to {WORKLOAD_DIR}")


if __name__ == "__main__":
    main()
