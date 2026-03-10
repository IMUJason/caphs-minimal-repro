from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from .models import PolicyConfig, ResourceSpec, ResourceState, TaskSpec, TaskState
from .policies import BasePolicy, build_policy


RESOURCE_BLUEPRINTS: list[dict[str, Any]] = [
    {"resource_id": "cpu0", "resource_type": "cpu", "speed": 1.0, "power_base": 8.0, "power_dynamic": 6.0, "heat_gain": 1.6, "cooling_rate": 0.12, "bandwidth_capacity": 8.5},
    {"resource_id": "cpu1", "resource_type": "cpu", "speed": 1.0, "power_base": 8.0, "power_dynamic": 6.0, "heat_gain": 1.6, "cooling_rate": 0.12, "bandwidth_capacity": 8.5},
    {"resource_id": "cpu2", "resource_type": "cpu", "speed": 1.0, "power_base": 8.0, "power_dynamic": 6.0, "heat_gain": 1.6, "cooling_rate": 0.12, "bandwidth_capacity": 8.5},
    {"resource_id": "cpu3", "resource_type": "cpu", "speed": 1.0, "power_base": 8.0, "power_dynamic": 6.0, "heat_gain": 1.6, "cooling_rate": 0.12, "bandwidth_capacity": 8.5},
    {"resource_id": "gpu0", "resource_type": "gpu", "speed": 2.6, "power_base": 18.0, "power_dynamic": 14.0, "heat_gain": 6.4, "cooling_rate": 0.04, "bandwidth_capacity": 14.0},
    {"resource_id": "gpu1", "resource_type": "gpu", "speed": 2.6, "power_base": 18.0, "power_dynamic": 14.0, "heat_gain": 6.4, "cooling_rate": 0.04, "bandwidth_capacity": 14.0},
    {"resource_id": "npu0", "resource_type": "npu", "speed": 2.1, "power_base": 10.0, "power_dynamic": 7.0, "heat_gain": 4.6, "cooling_rate": 0.05, "bandwidth_capacity": 10.0},
]


@dataclass
class SimulationResult:
    summary: dict[str, Any]
    events: pd.DataFrame
    timeline: pd.DataFrame


class Simulation:
    def __init__(self, workload_payload: dict[str, Any], policy_name: str, seed: int):
        self.metadata = workload_payload["metadata"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.policy: BasePolicy = build_policy(policy_name)
        self.tasks = self._build_tasks(workload_payload["tasks"])
        self.resources = self._build_resources()
        self.graph = self._build_graph()
        self.topological_order = list(nx.topological_sort(self.graph))
        self.current_time = 0
        self.events: list[dict[str, Any]] = []
        self.timeline: list[dict[str, Any]] = []
        self.monitor_overhead = 0.0
        self.scheduler_overhead = 0.0
        self.state_overhead = 0.0
        self.migration_benefit = 0.0
        self.migration_cost = 0.0
        self.policy.on_simulation_start(self)

    def _build_tasks(self, raw_tasks: list[dict[str, Any]]) -> dict[str, TaskState]:
        tasks: dict[str, TaskState] = {}
        for item in raw_tasks:
            spec = TaskSpec(
                task_id=item["task_id"],
                dag_id=item["dag_id"],
                arrival_time=int(item["arrival_time"]),
                priority=int(item["priority"]),
                latency_class=str(item["latency_class"]),
                task_type=str(item["task_type"]),
                size=float(item["size"]),
                bandwidth_demand=float(item["bandwidth_demand"]),
                checkpoint_interval=float(item["checkpoint_interval"]),
                migration_penalty=int(item["migration_penalty"]),
                service_time={key: int(value) for key, value in item["service_time"].items()},
                power_draw={key: float(value) for key, value in item["power_draw"].items()},
                deps=list(item["deps"]),
            )
            tasks[spec.task_id] = TaskState(spec=spec)
        return tasks

    def _build_resources(self) -> dict[str, ResourceState]:
        resources: dict[str, ResourceState] = {}
        for item in RESOURCE_BLUEPRINTS:
            spec = ResourceSpec(**item)
            resources[spec.resource_id] = ResourceState(spec=spec, temperature=spec.ambient_temp)
        return resources

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for task_id, task in self.tasks.items():
            graph.add_node(task_id)
            for parent in task.spec.deps:
                graph.add_edge(parent, task_id)
        return graph

    def ready_tasks(self) -> list[TaskState]:
        ready: list[TaskState] = []
        for task in self.tasks.values():
            if task.status != "pending":
                continue
            if task.spec.arrival_time > self.current_time:
                continue
            if any(self.tasks[parent].status != "completed" for parent in task.spec.deps):
                continue
            ready.append(task)
        return ready

    def running_tasks(self) -> list[TaskState]:
        return [task for task in self.tasks.values() if task.status == "running"]

    def migratable_tasks(self) -> list[TaskState]:
        tasks = []
        for task in self.running_tasks():
            if task.ready_for_checkpoint:
                tasks.append(task)
        return tasks

    def idle_resources(self) -> list[ResourceState]:
        return [resource for resource in self.resources.values() if resource.task_id is None]

    def type_snapshot(self) -> dict[str, Any]:
        predicted_load = {rtype: 0.0 for rtype in ("cpu", "gpu", "npu")}
        resource_counts = {rtype: 0 for rtype in ("cpu", "gpu", "npu")}
        mean_temperature = {rtype: 0.0 for rtype in ("cpu", "gpu", "npu")}
        bandwidth_pressure = {rtype: 0.0 for rtype in ("cpu", "gpu", "npu")}
        waiting_tasks = self.ready_tasks()
        for resource in self.resources.values():
            resource_type = resource.spec.resource_type
            resource_counts[resource_type] += 1
            mean_temperature[resource_type] += resource.temperature
            if resource.task_id is not None:
                task = self.tasks[resource.task_id]
                predicted_load[resource_type] += task.remaining_time_on(resource_type)
                bandwidth_pressure[resource_type] += task.spec.bandwidth_demand / resource.spec.bandwidth_capacity
        for task in waiting_tasks:
            best_type = min(task.spec.service_time, key=task.spec.service_time.get)
            predicted_load[best_type] += task.remaining_time_on(best_type) * 0.3
            bandwidth_pressure[best_type] += task.spec.bandwidth_demand / max(
                1.0,
                sum(
                    resource.spec.bandwidth_capacity
                    for resource in self.resources.values()
                    if resource.spec.resource_type == best_type
                ),
            )
        for resource_type, count in resource_counts.items():
            if count:
                mean_temperature[resource_type] /= count
        return {
            "predicted_load": predicted_load,
            "resource_counts": resource_counts,
            "mean_temperature": mean_temperature,
            "bandwidth_pressure": bandwidth_pressure,
        }

    def _assign_task(self, task: TaskState, resource: ResourceState, metadata: dict[str, Any], migrated: bool = False) -> None:
        resource.task_id = task.spec.task_id
        task.current_resource_id = resource.spec.resource_id
        task.status = "running"
        if task.start_time is None:
            task.start_time = self.current_time
        if migrated:
            resource.migration_delay_remaining = task.spec.migration_penalty
            task.migrations += 1
            task.cooldown_remaining = 0
            self.migration_cost += metadata.get("migration_cost", float(task.spec.migration_penalty))
        event = {
            "time": self.current_time,
            "event_type": "migration" if migrated else "dispatch",
            "task_id": task.spec.task_id,
            "dag_id": task.spec.dag_id,
            "resource_id": resource.spec.resource_id,
            "resource_type": resource.spec.resource_type,
            "policy": self.policy.name,
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }
        self.events.append(event)

    def _apply_decisions(self) -> None:
        monitor_start = time.perf_counter()
        self.type_snapshot()
        self.monitor_overhead += time.perf_counter() - monitor_start
        scheduler_start = time.perf_counter()
        decision = self.policy.decide(self, self.rng)
        self.scheduler_overhead += time.perf_counter() - scheduler_start
        self.state_overhead += getattr(self.policy, "last_state_update_overhead", 0.0)
        for task_id, source_resource_id, target_resource_id, metadata in decision.migrations:
            task = self.tasks[task_id]
            if task.current_resource_id != source_resource_id:
                continue
            source = self.resources[source_resource_id]
            target = self.resources[target_resource_id]
            if target.task_id is not None:
                continue
            current_type = source.spec.resource_type
            target_type = target.spec.resource_type
            current_remaining = task.remaining_time_on(current_type)
            target_remaining = task.remaining_time_on(target_type)
            self.migration_benefit += max(0.0, current_remaining - target_remaining)
            source.task_id = None
            task.last_checkpoint_fraction = task.completed_fraction
            task.cooldown_remaining = self.policy.config.cooldown_steps if hasattr(self.policy, "config") else 2
            self._assign_task(task, target, metadata, migrated=True)
        for task_id, resource_id, metadata in decision.dispatches:
            task = self.tasks[task_id]
            resource = self.resources[resource_id]
            if task.status != "pending" or resource.task_id is not None:
                continue
            self._assign_task(task, resource, metadata, migrated=False)

    def _advance_resources(self) -> None:
        for task in self.tasks.values():
            if task.cooldown_remaining > 0:
                task.cooldown_remaining -= 1
        total_power = 0.0
        overall_bandwidth = 0.0
        completed_tasks_this_step = 0
        for resource in self.resources.values():
            if resource.task_id is None:
                cooling = resource.spec.cooling_rate * (resource.temperature - resource.spec.ambient_temp)
                resource.temperature = max(resource.spec.ambient_temp, resource.temperature - cooling)
                continue
            task = self.tasks[resource.task_id]
            total_power += resource.spec.power_base + task.spec.power_draw[resource.spec.resource_type]
            overall_bandwidth += task.spec.bandwidth_demand
            if resource.migration_delay_remaining > 0:
                resource.migration_delay_remaining -= 1
            else:
                decrement = 1.0 / max(1, task.spec.service_time[resource.spec.resource_type])
                task.remaining_fraction = max(0.0, task.remaining_fraction - decrement)
                task.progress_history.append(
                    {
                        "time": self.current_time,
                        "resource_id": resource.spec.resource_id,
                        "resource_type": resource.spec.resource_type,
                        "remaining_fraction": round(task.remaining_fraction, 6),
                    }
                )
            resource.busy_steps += 1
            utilization_boost = min(1.5, task.spec.bandwidth_demand / resource.spec.bandwidth_capacity)
            cooling = resource.spec.cooling_rate * (resource.temperature - resource.spec.ambient_temp)
            resource.temperature = resource.temperature + resource.spec.heat_gain * utilization_boost - cooling
            if task.remaining_fraction <= 1e-9:
                task.status = "completed"
                task.completion_time = self.current_time + 1
                task.current_resource_id = None
                resource.task_id = None
                resource.migration_delay_remaining = 0
                completed_tasks_this_step += 1
                self.events.append(
                    {
                        "time": self.current_time + 1,
                        "event_type": "complete",
                        "task_id": task.spec.task_id,
                        "dag_id": task.spec.dag_id,
                        "resource_id": resource.spec.resource_id,
                        "resource_type": resource.spec.resource_type,
                        "policy": self.policy.name,
                        "metadata_json": "{}",
                    }
                )
        self.timeline.append(
            {
                "time": self.current_time,
                "policy": self.policy.name,
                "ready_tasks": len(self.ready_tasks()),
                "running_tasks": len(self.running_tasks()),
                "completed_tasks": sum(task.status == "completed" for task in self.tasks.values()),
                "completed_tasks_this_step": completed_tasks_this_step,
                "total_power": round(total_power, 4),
                "overall_bandwidth": round(overall_bandwidth, 4),
                "cpu_util": self._utilization("cpu"),
                "gpu_util": self._utilization("gpu"),
                "npu_util": self._utilization("npu"),
                "cpu_temp": round(self._mean_temperature("cpu"), 4),
                "gpu_temp": round(self._mean_temperature("gpu"), 4),
                "npu_temp": round(self._mean_temperature("npu"), 4),
            }
        )

    def _utilization(self, resource_type: str) -> float:
        resources = [resource for resource in self.resources.values() if resource.spec.resource_type == resource_type]
        busy = sum(resource.task_id is not None for resource in resources)
        return busy / max(1, len(resources))

    def _mean_temperature(self, resource_type: str) -> float:
        resources = [resource for resource in self.resources.values() if resource.spec.resource_type == resource_type]
        if not resources:
            return 0.0
        return float(np.mean([resource.temperature for resource in resources]))

    def run(self, max_time: int = 5000) -> SimulationResult:
        while self.current_time < max_time and not all(task.status == "completed" for task in self.tasks.values()):
            self._apply_decisions()
            self._advance_resources()
            self.current_time += 1
        summary = self._build_summary()
        return SimulationResult(summary=summary, events=pd.DataFrame(self.events), timeline=pd.DataFrame(self.timeline))

    def _build_summary(self) -> dict[str, Any]:
        completions = np.array([task.completion_time for task in self.tasks.values()], dtype=float)
        arrivals = np.array([task.spec.arrival_time for task in self.tasks.values()], dtype=float)
        latencies = completions - arrivals
        total_time = max(1.0, float(np.max(completions)))
        temps = np.concatenate(
            [
                np.array([resource.temperature for resource in self.resources.values()]),
                self.timeline_frame()[["cpu_temp", "gpu_temp", "npu_temp"]].to_numpy().flatten(),
            ]
        )
        energy = float(self.timeline_frame()["total_power"].sum())
        migration_count = int(sum(task.migrations for task in self.tasks.values()))
        migration_ratio = self.migration_benefit / max(1e-9, self.migration_cost) if migration_count else 0.0
        cpu_util = float(self.timeline_frame()["cpu_util"].mean())
        gpu_util = float(self.timeline_frame()["gpu_util"].mean())
        npu_util = float(self.timeline_frame()["npu_util"].mean())
        load_balance = 1.0 - float(np.std([cpu_util, gpu_util, npu_util]) / max(1e-6, np.mean([cpu_util, gpu_util, npu_util])))
        return {
            "policy": self.policy.name,
            "workload_name": self.metadata["workload_name"],
            "workload_seed": self.metadata["seed"],
            "simulation_seed": self.seed,
            "task_count": len(self.tasks),
            "makespan": round(total_time, 4),
            "throughput": round(len(self.tasks) / total_time, 6),
            "average_latency": round(float(np.mean(latencies)), 4),
            "p95_latency": round(float(np.percentile(latencies, 95)), 4),
            "p99_latency": round(float(np.percentile(latencies, 99)), 4),
            "cpu_utilization": round(cpu_util, 6),
            "gpu_utilization": round(gpu_util, 6),
            "npu_utilization": round(npu_util, 6),
            "load_balance_index": round(load_balance, 6),
            "number_of_migrations": migration_count,
            "migration_success_rate": round(1.0 if migration_count else 0.0, 4),
            "migration_benefit_ratio": round(float(migration_ratio), 6),
            "migration_overhead_time": round(float(self.migration_cost), 4),
            "total_energy": round(energy, 4),
            "average_power": round(float(self.timeline_frame()["total_power"].mean()), 4),
            "peak_power": round(float(self.timeline_frame()["total_power"].max()), 4),
            "peak_temperature": round(float(np.max(temps)), 4),
            "temperature_variance": round(float(np.var(temps)), 6),
            "time_above_thermal_threshold": int((temps > 90.0).sum()),
            "scheduler_decision_latency_ms": round(self.scheduler_overhead * 1000, 4),
            "monitoring_overhead_ms": round(self.monitor_overhead * 1000, 4),
            "state_update_overhead_ms": round(self.state_overhead * 1000, 4),
        }

    def timeline_frame(self) -> pd.DataFrame:
        if not self.timeline:
            return pd.DataFrame(
                [
                    {
                        "time": 0,
                        "total_power": 0.0,
                        "cpu_util": 0.0,
                        "gpu_util": 0.0,
                        "npu_util": 0.0,
                        "cpu_temp": 40.0,
                        "gpu_temp": 40.0,
                        "npu_temp": 40.0,
                    }
                ]
            )
        return pd.DataFrame(self.timeline)


def simulate_dataset(workload_path: str | Path, policy_name: str, seed: int) -> SimulationResult:
    payload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    simulation = Simulation(payload, policy_name=policy_name, seed=seed)
    return simulation.run()
