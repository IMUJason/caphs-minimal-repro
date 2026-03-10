from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .models import PolicyConfig, RESOURCE_TYPES, ResourceState, TaskState


def _softmax(values: list[float], tau: float) -> list[float]:
    array = np.array(values, dtype=float)
    if tau <= 0:
        tau = 1e-6
    scaled = array / tau
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    return probs.tolist()


def _rank_score(task: TaskState) -> float:
    return float(task.spec.priority * 10 - task.spec.arrival_time)


def _average_service(task: TaskState) -> float:
    return float(np.mean([task.spec.service_time[key] for key in RESOURCE_TYPES]))


@dataclass
class PolicyDecision:
    dispatches: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    migrations: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)


class BasePolicy:
    def __init__(self, name: str):
        self.name = name

    def on_simulation_start(self, simulation: Any) -> None:
        return None

    def decide(self, simulation: Any, rng: np.random.Generator) -> PolicyDecision:
        return PolicyDecision()


class SimpleDispatchPolicy(BasePolicy):
    def __init__(self, name: str):
        super().__init__(name)

    def ordered_ready_tasks(self, simulation: Any) -> list[TaskState]:
        return sorted(simulation.ready_tasks(), key=lambda task: (task.spec.arrival_time, -task.spec.priority, task.spec.task_id))

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        if not idle_resources:
            return None, {}
        selected = min(idle_resources, key=lambda resource: resource.spec.resource_id)
        return selected, {}

    def decide(self, simulation: Any, rng: np.random.Generator) -> PolicyDecision:
        idle_resources = simulation.idle_resources()
        ready_tasks = self.ordered_ready_tasks(simulation)
        decision = PolicyDecision()
        for task in ready_tasks:
            if not idle_resources:
                break
            resource, features = self.select_resource(simulation, task, idle_resources)
            if resource is None:
                continue
            decision.dispatches.append((task.spec.task_id, resource.spec.resource_id, features))
            idle_resources = [item for item in idle_resources if item.spec.resource_id != resource.spec.resource_id]
        return decision


class FCFSPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("fcfs")


class FixedPriorityPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("fixed_priority")

    def ordered_ready_tasks(self, simulation: Any) -> list[TaskState]:
        return sorted(
            simulation.ready_tasks(),
            key=lambda task: (-task.spec.priority, task.spec.arrival_time, task.spec.task_id),
        )

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        resource = min(
            idle_resources,
            key=lambda candidate: (
                task.remaining_time_on(candidate.spec.resource_type),
                candidate.temperature,
                candidate.spec.resource_id,
            ),
        )
        return resource, {"criterion": "priority_then_fastest"}


class LeastLoadedPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("least_loaded")

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        snapshot = simulation.type_snapshot()
        resource = min(
            idle_resources,
            key=lambda candidate: (
                snapshot["predicted_load"][candidate.spec.resource_type],
                task.remaining_time_on(candidate.spec.resource_type),
                candidate.temperature,
            ),
        )
        return resource, {"criterion": "predicted_load"}


class WorkStealingPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("work_stealing")

    def ordered_ready_tasks(self, simulation: Any) -> list[TaskState]:
        affinity_sorted = sorted(
            simulation.ready_tasks(),
            key=lambda task: (
                min(task.spec.service_time.values()),
                -task.spec.priority,
                task.spec.arrival_time,
            ),
        )
        return affinity_sorted

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        preferred_type = min(task.spec.service_time, key=task.spec.service_time.get)
        same_type = [resource for resource in idle_resources if resource.spec.resource_type == preferred_type]
        if same_type:
            resource = min(same_type, key=lambda candidate: candidate.temperature)
            return resource, {"criterion": "local_queue"}
        resource = min(
            idle_resources,
            key=lambda candidate: (
                task.remaining_time_on(candidate.spec.resource_type),
                candidate.temperature,
            ),
        )
        return resource, {"criterion": "steal"}


class HEFTPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("heft")
        self.upward_ranks: dict[str, float] = {}

    def on_simulation_start(self, simulation: Any) -> None:
        graph = simulation.graph
        reverse_order = list(reversed(list(simulation.topological_order)))
        for task_id in reverse_order:
            task = simulation.tasks[task_id]
            successors = list(graph.successors(task_id))
            if not successors:
                self.upward_ranks[task_id] = _average_service(task)
                continue
            self.upward_ranks[task_id] = _average_service(task) + max(
                self.upward_ranks[child] for child in successors
            )

    def ordered_ready_tasks(self, simulation: Any) -> list[TaskState]:
        return sorted(
            simulation.ready_tasks(),
            key=lambda task: (
                -self.upward_ranks.get(task.spec.task_id, _average_service(task)),
                -task.spec.priority,
                task.spec.arrival_time,
            ),
        )

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        snapshot = simulation.type_snapshot()
        resource = min(
            idle_resources,
            key=lambda candidate: (
                task.remaining_time_on(candidate.spec.resource_type)
                + snapshot["predicted_load"][candidate.spec.resource_type],
                candidate.temperature,
            ),
        )
        return resource, {"criterion": "earliest_finish_time"}


class FeedbackThresholdPolicy(SimpleDispatchPolicy):
    def __init__(self) -> None:
        super().__init__("feedback_threshold")

    def select_resource(self, simulation: Any, task: TaskState, idle_resources: list[ResourceState]) -> tuple[ResourceState | None, dict[str, Any]]:
        snapshot = simulation.type_snapshot()
        best_resource = None
        best_score = -math.inf
        best_features: dict[str, Any] = {}
        for resource in idle_resources:
            resource_type = resource.spec.resource_type
            queue_penalty = snapshot["predicted_load"][resource_type] / max(1.0, snapshot["resource_counts"][resource_type])
            exec_penalty = task.remaining_time_on(resource_type)
            thermal_penalty = max(0.0, resource.temperature - (resource.spec.thermal_limit - 8.0)) / 10.0
            score = (
                4.0 / exec_penalty
                + task.spec.priority * 0.2
                - queue_penalty * 0.08
                - thermal_penalty * 0.9
            )
            if score > best_score:
                best_resource = resource
                best_score = score
                best_features = {
                    "criterion": "feedback_threshold",
                    "score": round(score, 4),
                    "queue_penalty": round(queue_penalty, 4),
                    "thermal_penalty": round(thermal_penalty, 4),
                }
        return best_resource, best_features


class ProposedStateEvolutionPolicy(BasePolicy):
    def __init__(self, name: str, config: PolicyConfig):
        super().__init__(name)
        self.config = config
        self.state_scores: dict[str, dict[str, float]] = {}
        self.last_state_update_overhead = 0.0

    def _ensure_state(self, task: TaskState) -> None:
        if task.spec.task_id not in self.state_scores:
            initial = {rtype: 0.0 for rtype in RESOURCE_TYPES}
            for resource_type in RESOURCE_TYPES:
                initial[resource_type] = 1.0 / task.spec.service_time[resource_type]
            self.state_scores[task.spec.task_id] = initial

    def _type_features(self, simulation: Any, task: TaskState, resource_type: str) -> dict[str, float]:
        snapshot = simulation.type_snapshot()
        affinity = 1.0 / task.remaining_time_on(resource_type)
        queue_delay = snapshot["predicted_load"][resource_type] / max(1, snapshot["resource_counts"][resource_type])
        exec_penalty = task.remaining_time_on(resource_type)
        migration_cost = 0.0
        if task.current_resource_id is not None:
            current_type = simulation.resources[task.current_resource_id].spec.resource_type
            if current_type != resource_type:
                migration_cost = task.spec.migration_penalty + task.spec.bandwidth_demand * 0.5
        power_penalty = task.spec.power_draw[resource_type]
        thermal_penalty = snapshot["mean_temperature"][resource_type] / 20.0
        bandwidth_penalty = snapshot["bandwidth_pressure"][resource_type]
        return {
            "affinity": affinity,
            "queue_delay": queue_delay,
            "exec_penalty": exec_penalty,
            "migration_cost": migration_cost,
            "power_penalty": power_penalty,
            "thermal_penalty": thermal_penalty,
            "bandwidth_penalty": bandwidth_penalty,
        }

    def _update_scores(self, simulation: Any, task: TaskState) -> dict[str, float]:
        start = time.perf_counter()
        self._ensure_state(task)
        previous = self.state_scores[task.spec.task_id]
        updated: dict[str, float] = {}
        for resource_type in RESOURCE_TYPES:
            features = self._type_features(simulation, task, resource_type)
            immediate = (
                4.8 * features["affinity"]
                + task.spec.priority * 0.08
                - self.config.queue_weight * features["queue_delay"]
                - self.config.exec_weight * features["exec_penalty"]
                - self.config.migration_weight * features["migration_cost"]
                - self.config.energy_weight * features["power_penalty"]
                - self.config.thermal_weight * features["thermal_penalty"]
                - self.config.bandwidth_weight * features["bandwidth_penalty"]
            )
            if self.config.feedback_enabled:
                updated[resource_type] = (1.0 - self.config.eta) * previous[resource_type] + self.config.eta * immediate
            else:
                updated[resource_type] = immediate
        self.state_scores[task.spec.task_id] = updated
        self.last_state_update_overhead += time.perf_counter() - start
        return updated

    def _pick_type(
        self,
        scores: dict[str, float],
        available_types: list[str],
        rng: np.random.Generator,
    ) -> tuple[str, dict[str, Any]]:
        filtered_scores = [scores[resource_type] for resource_type in available_types]
        probabilities = _softmax(filtered_scores, self.config.tau)
        if self.config.probabilistic:
            index = int(rng.choice(len(available_types), p=probabilities))
        else:
            index = int(np.argmax(probabilities))
        return available_types[index], {
            "available_types": available_types,
            "score_vector": {rtype: round(scores[rtype], 4) for rtype in available_types},
            "prob_vector": {rtype: round(prob, 4) for rtype, prob in zip(available_types, probabilities)},
        }

    def _choose_instance(self, idle_resources: list[ResourceState], resource_type: str) -> ResourceState:
        candidates = [resource for resource in idle_resources if resource.spec.resource_type == resource_type]
        return min(candidates, key=lambda resource: (resource.temperature, resource.spec.resource_id))

    def _dispatch_decisions(self, simulation: Any, rng: np.random.Generator) -> list[tuple[str, str, dict[str, Any]]]:
        ready_tasks = sorted(
            simulation.ready_tasks(),
            key=lambda task: (-_rank_score(task), task.spec.task_id),
        )
        idle_resources = simulation.idle_resources()
        dispatches: list[tuple[str, str, dict[str, Any]]] = []
        for task in ready_tasks:
            if not idle_resources:
                break
            scores = self._update_scores(simulation, task)
            available_types = sorted({resource.spec.resource_type for resource in idle_resources})
            chosen_type, metadata = self._pick_type(scores, available_types, rng)
            resource = self._choose_instance(idle_resources, chosen_type)
            features = self._type_features(simulation, task, chosen_type)
            metadata.update(
                {
                    "criterion": "state_evolution",
                    "features": {key: round(value, 4) for key, value in features.items()},
                }
            )
            dispatches.append((task.spec.task_id, resource.spec.resource_id, metadata))
            idle_resources = [item for item in idle_resources if item.spec.resource_id != resource.spec.resource_id]
        return dispatches

    def _migration_decisions(self, simulation: Any, rng: np.random.Generator) -> list[tuple[str, str, str, dict[str, Any]]]:
        if not self.config.migration_enabled:
            return []
        idle_resources = simulation.idle_resources()
        if not idle_resources:
            return []
        decisions: list[tuple[str, str, str, dict[str, Any]]] = []
        used_resources: set[str] = set()
        for task in simulation.migratable_tasks():
            if task.cooldown_remaining > 0:
                continue
            current_resource = simulation.resources[task.current_resource_id]
            scores = self._update_scores(simulation, task)
            probabilities = _softmax([scores[key] for key in RESOURCE_TYPES], self.config.tau)
            prob_map = {rtype: prob for rtype, prob in zip(RESOURCE_TYPES, probabilities)}
            best_type = max(RESOURCE_TYPES, key=lambda resource_type: scores[resource_type])
            if best_type == current_resource.spec.resource_type:
                continue
            if prob_map[best_type] < self.config.migration_threshold:
                continue
            candidates = [
                resource
                for resource in idle_resources
                if resource.spec.resource_type == best_type and resource.spec.resource_id not in used_resources
            ]
            if not candidates:
                continue
            target = min(candidates, key=lambda resource: (resource.temperature, resource.spec.resource_id))
            current_remaining = task.remaining_time_on(current_resource.spec.resource_type)
            target_remaining = task.remaining_time_on(best_type)
            migration_cost = task.spec.migration_penalty + task.spec.bandwidth_demand * 0.5
            net_gain = current_remaining - target_remaining - migration_cost
            thermal_safe = target.temperature <= target.spec.thermal_limit - self.config.thermal_safety_margin
            if net_gain <= self.config.net_gain_threshold or not thermal_safe:
                continue
            metadata = {
                "criterion": "state_evolution_migration",
                "score_vector": {rtype: round(scores[rtype], 4) for rtype in RESOURCE_TYPES},
                "prob_vector": {rtype: round(prob_map[rtype], 4) for rtype in RESOURCE_TYPES},
                "net_gain": round(float(net_gain), 4),
                "migration_cost": round(float(migration_cost), 4),
            }
            decisions.append((task.spec.task_id, current_resource.spec.resource_id, target.spec.resource_id, metadata))
            used_resources.add(target.spec.resource_id)
        return decisions

    def decide(self, simulation: Any, rng: np.random.Generator) -> PolicyDecision:
        self.last_state_update_overhead = 0.0
        return PolicyDecision(
            dispatches=self._dispatch_decisions(simulation, rng),
            migrations=self._migration_decisions(simulation, rng),
        )


class PredictiveLyapunovPolicy(ProposedStateEvolutionPolicy):
    LATENCY_CLASS_FACTOR = {
        "interactive": 1.25,
        "latency_sensitive": 1.2,
        "throughput_oriented": 1.0,
        "best_effort": 0.85,
    }

    def __init__(self, name: str, config: PolicyConfig):
        super().__init__(name, config)
        self.task_criticality: dict[str, float] = {}
        self.thermal_targets: dict[str, float] = {rtype: 80.0 for rtype in RESOURCE_TYPES}
        self.thermal_virtual_queues: dict[str, float] = {rtype: 0.0 for rtype in RESOURCE_TYPES}
        self.bandwidth_virtual_queues: dict[str, float] = {rtype: 0.0 for rtype in RESOURCE_TYPES}
        self.migration_virtual_queue = 0.0
        self.last_snapshot: dict[str, Any] | None = None

    def on_simulation_start(self, simulation: Any) -> None:
        self._initialize_criticality(simulation)
        self._initialize_thermal_targets(simulation)

    def _initialize_criticality(self, simulation: Any) -> None:
        best_service = {
            task_id: min(task.spec.service_time.values())
            for task_id, task in simulation.tasks.items()
        }
        critical_path: dict[str, float] = {}
        descendants: dict[str, int] = {}
        for task_id in reversed(simulation.topological_order):
            children = list(simulation.graph.successors(task_id))
            critical_path[task_id] = best_service[task_id] + (
                max((critical_path[child] for child in children), default=0.0)
            )
            descendants[task_id] = len(nx.descendants(simulation.graph, task_id))
        max_path = max(critical_path.values(), default=1.0)
        max_descendants = max(descendants.values(), default=1)
        self.task_criticality = {
            task_id: 0.75 * (critical_path[task_id] / max_path)
            + 0.25 * (descendants[task_id] / max_descendants if max_descendants else 0.0)
            for task_id in simulation.tasks
        }

    def _initialize_thermal_targets(self, simulation: Any) -> None:
        for resource_type in RESOURCE_TYPES:
            resources = [
                resource
                for resource in simulation.resources.values()
                if resource.spec.resource_type == resource_type
            ]
            if not resources:
                continue
            ambient = float(np.mean([resource.spec.ambient_temp for resource in resources]))
            thermal_limit = float(np.mean([resource.spec.thermal_limit for resource in resources]))
            self.thermal_targets[resource_type] = ambient + self.config.thermal_target_ratio * (thermal_limit - ambient)

    def _latency_factor(self, task: TaskState) -> float:
        return self.LATENCY_CLASS_FACTOR.get(task.spec.latency_class, 1.0)

    def _age_bonus(self, simulation: Any, task: TaskState) -> float:
        age = max(0, simulation.current_time - task.spec.arrival_time)
        return math.log1p(age) / 4.0

    def _task_rank(self, simulation: Any, task: TaskState) -> float:
        latency_factor = self._latency_factor(task)
        criticality = self.task_criticality.get(task.spec.task_id, 0.0)
        age_bonus = self._age_bonus(simulation, task)
        return (
            _rank_score(task)
            + criticality * 8.0 * latency_factor
            + age_bonus * 2.5
        )

    def _refresh_virtual_queues(self, simulation: Any) -> dict[str, Any]:
        snapshot = simulation.type_snapshot()
        for resource_type in RESOURCE_TYPES:
            thermal_excess = max(
                0.0,
                (snapshot["mean_temperature"][resource_type] - self.thermal_targets[resource_type]) / 8.0,
            )
            self.thermal_virtual_queues[resource_type] = max(
                0.0,
                self.config.virtual_queue_decay * self.thermal_virtual_queues[resource_type] + thermal_excess,
            )
            bandwidth_excess = max(0.0, snapshot["bandwidth_pressure"][resource_type] - self.config.bandwidth_target)
            self.bandwidth_virtual_queues[resource_type] = max(
                0.0,
                self.config.virtual_queue_decay * self.bandwidth_virtual_queues[resource_type] + bandwidth_excess,
            )
        active_migrations = sum(task.cooldown_remaining > 0 for task in simulation.tasks.values())
        self.migration_virtual_queue = max(
            0.0,
            self.config.virtual_queue_decay * self.migration_virtual_queue
            + max(0.0, active_migrations - self.config.migration_budget),
        )
        self.last_snapshot = snapshot
        return snapshot

    def _type_features(
        self,
        simulation: Any,
        task: TaskState,
        resource_type: str,
        snapshot: dict[str, Any] | None = None,
        extra_load: dict[str, float] | None = None,
    ) -> dict[str, float]:
        snapshot = snapshot or self.last_snapshot or simulation.type_snapshot()
        extra_load = extra_load or {rtype: 0.0 for rtype in RESOURCE_TYPES}
        affinity = 1.0 / task.remaining_time_on(resource_type)
        queue_delay = (
            snapshot["predicted_load"][resource_type] + extra_load.get(resource_type, 0.0)
        ) / max(1, snapshot["resource_counts"][resource_type])
        exec_penalty = task.remaining_time_on(resource_type)
        predicted_completion = queue_delay + exec_penalty
        migration_cost = 0.0
        if task.current_resource_id is not None:
            current_type = simulation.resources[task.current_resource_id].spec.resource_type
            if current_type != resource_type:
                migration_cost = task.spec.migration_penalty + task.spec.bandwidth_demand * 0.5
        power_penalty = task.spec.power_draw[resource_type]
        mean_temperature = snapshot["mean_temperature"][resource_type]
        thermal_penalty = max(0.0, mean_temperature - self.thermal_targets[resource_type]) / 10.0
        thermal_slack = max(0.0, self.thermal_targets[resource_type] - mean_temperature) / 10.0
        bandwidth_penalty = snapshot["bandwidth_pressure"][resource_type]
        return {
            "affinity": affinity,
            "queue_delay": queue_delay,
            "exec_penalty": exec_penalty,
            "predicted_completion": predicted_completion,
            "migration_cost": migration_cost,
            "power_penalty": power_penalty,
            "thermal_penalty": thermal_penalty,
            "thermal_slack": thermal_slack,
            "bandwidth_penalty": bandwidth_penalty,
            "criticality": self.task_criticality.get(task.spec.task_id, 0.0),
            "age_bonus": self._age_bonus(simulation, task),
            "latency_factor": self._latency_factor(task),
            "thermal_virtual": self.thermal_virtual_queues[resource_type],
            "bandwidth_virtual": self.bandwidth_virtual_queues[resource_type],
        }

    def _update_scores(
        self,
        simulation: Any,
        task: TaskState,
        snapshot: dict[str, Any] | None = None,
        extra_load: dict[str, float] | None = None,
    ) -> dict[str, float]:
        start = time.perf_counter()
        self._ensure_state(task)
        snapshot = snapshot or self.last_snapshot or self._refresh_virtual_queues(simulation)
        previous = self.state_scores[task.spec.task_id]
        updated: dict[str, float] = {}
        for resource_type in RESOURCE_TYPES:
            features = self._type_features(
                simulation,
                task,
                resource_type,
                snapshot=snapshot,
                extra_load=extra_load,
            )
            urgency = (
                task.spec.priority * 0.10 * features["latency_factor"]
                + self.config.criticality_weight * features["criticality"] * features["latency_factor"]
                + self.config.starvation_weight * features["age_bonus"]
            )
            immediate = (
                4.8 * features["affinity"]
                + urgency
                + self.config.slack_weight * features["thermal_slack"]
                - self.config.queue_forecast_weight * features["queue_delay"]
                - self.config.completion_weight * features["predicted_completion"]
                - self.config.exec_weight * features["exec_penalty"]
                - self.config.migration_weight
                * (features["migration_cost"] + self.config.migration_virtual_weight * self.migration_virtual_queue)
                - self.config.energy_weight * features["power_penalty"]
                - self.config.bandwidth_weight
                * features["bandwidth_penalty"]
                * (1.0 + self.config.bandwidth_virtual_weight * features["bandwidth_virtual"])
                - self.config.thermal_weight
                * features["thermal_penalty"]
                * (1.0 + self.config.thermal_virtual_weight * features["thermal_virtual"])
            )
            updated[resource_type] = (1.0 - self.config.eta) * previous[resource_type] + self.config.eta * immediate
        self.state_scores[task.spec.task_id] = updated
        self.last_state_update_overhead += time.perf_counter() - start
        return updated

    def _dispatch_decisions(self, simulation: Any, rng: np.random.Generator) -> list[tuple[str, str, dict[str, Any]]]:
        snapshot = self._refresh_virtual_queues(simulation)
        ready_tasks = sorted(
            simulation.ready_tasks(),
            key=lambda task: (-self._task_rank(simulation, task), task.spec.task_id),
        )
        idle_resources = simulation.idle_resources()
        extra_load = {rtype: 0.0 for rtype in RESOURCE_TYPES}
        dispatches: list[tuple[str, str, dict[str, Any]]] = []
        for task in ready_tasks:
            if not idle_resources:
                break
            scores = self._update_scores(simulation, task, snapshot=snapshot, extra_load=extra_load)
            available_types = sorted({resource.spec.resource_type for resource in idle_resources})
            chosen_type, metadata = self._pick_type(scores, available_types, rng)
            resource = self._choose_instance(idle_resources, chosen_type)
            features = self._type_features(
                simulation,
                task,
                chosen_type,
                snapshot=snapshot,
                extra_load=extra_load,
            )
            metadata.update(
                {
                    "criterion": "predictive_lyapunov_dispatch",
                    "task_rank": round(self._task_rank(simulation, task), 4),
                    "migration_virtual_queue": round(self.migration_virtual_queue, 4),
                    "thermal_virtual_queues": {
                        rtype: round(self.thermal_virtual_queues[rtype], 4) for rtype in RESOURCE_TYPES
                    },
                    "bandwidth_virtual_queues": {
                        rtype: round(self.bandwidth_virtual_queues[rtype], 4) for rtype in RESOURCE_TYPES
                    },
                    "features": {key: round(value, 4) for key, value in features.items()},
                }
            )
            dispatches.append((task.spec.task_id, resource.spec.resource_id, metadata))
            extra_load[chosen_type] += task.remaining_time_on(chosen_type)
            idle_resources = [item for item in idle_resources if item.spec.resource_id != resource.spec.resource_id]
        return dispatches

    def _migration_decisions(self, simulation: Any, rng: np.random.Generator) -> list[tuple[str, str, str, dict[str, Any]]]:
        if not self.config.migration_enabled:
            return []
        idle_resources = simulation.idle_resources()
        if not idle_resources:
            return []
        snapshot = self.last_snapshot or self._refresh_virtual_queues(simulation)
        candidate_migrations: list[tuple[float, tuple[str, str, str, dict[str, Any]]]] = []
        for task in sorted(
            simulation.migratable_tasks(),
            key=lambda item: (
                self.task_criticality.get(item.spec.task_id, 0.0),
                item.spec.priority,
                self._age_bonus(simulation, item),
            ),
            reverse=True,
        ):
            if task.cooldown_remaining > 0:
                continue
            current_resource = simulation.resources[task.current_resource_id]
            current_type = current_resource.spec.resource_type
            scores = self._update_scores(simulation, task, snapshot=snapshot)
            current_score = scores[current_type]
            best_gain = -math.inf
            best_choice: tuple[str, str, str, dict[str, Any]] | None = None
            for resource_type in sorted({resource.spec.resource_type for resource in idle_resources}):
                if resource_type == current_type:
                    continue
                candidates = [resource for resource in idle_resources if resource.spec.resource_type == resource_type]
                if not candidates:
                    continue
                target = min(candidates, key=lambda resource: (resource.temperature, resource.spec.resource_id))
                target_features = self._type_features(simulation, task, resource_type, snapshot=snapshot)
                current_features = self._type_features(simulation, task, current_type, snapshot=snapshot)
                score_margin = scores[resource_type] - current_score
                net_gain = (
                    current_features["exec_penalty"]
                    - target_features["exec_penalty"]
                    - target_features["migration_cost"]
                )
                thermal_relief = max(0.0, current_resource.temperature - target.temperature) / 10.0
                composite_gain = (
                    net_gain
                    + 0.25 * score_margin
                    + 0.35 * thermal_relief
                    + 0.15 * self.task_criticality.get(task.spec.task_id, 0.0)
                )
                threshold = (
                    self.config.net_gain_threshold
                    + 0.18 * self.migration_virtual_queue
                    + 0.06 * self.bandwidth_virtual_queues[resource_type]
                )
                thermal_safe = target.temperature <= target.spec.thermal_limit - self.config.thermal_safety_margin
                if score_margin <= 0.04 or composite_gain <= threshold or not thermal_safe:
                    continue
                metadata = {
                    "criterion": "predictive_lyapunov_migration",
                    "score_vector": {rtype: round(scores[rtype], 4) for rtype in RESOURCE_TYPES},
                    "score_margin": round(score_margin, 4),
                    "net_gain": round(float(net_gain), 4),
                    "thermal_relief": round(float(thermal_relief), 4),
                    "composite_gain": round(float(composite_gain), 4),
                    "migration_cost": round(float(target_features["migration_cost"]), 4),
                    "migration_virtual_queue": round(self.migration_virtual_queue, 4),
                }
                if composite_gain > best_gain:
                    best_gain = composite_gain
                    best_choice = (
                        task.spec.task_id,
                        current_resource.spec.resource_id,
                        target.spec.resource_id,
                        metadata,
                    )
            if best_choice is not None:
                candidate_migrations.append((best_gain, best_choice))
        candidate_migrations.sort(key=lambda item: item[0], reverse=True)
        decisions: list[tuple[str, str, str, dict[str, Any]]] = []
        used_targets: set[str] = set()
        for _, choice in candidate_migrations:
            _, _, target_resource_id, _ = choice
            if target_resource_id in used_targets:
                continue
            decisions.append(choice)
            used_targets.add(target_resource_id)
        return decisions


def build_policy(name: str) -> BasePolicy:
    normalized = name.lower()
    if normalized == "fcfs":
        return FCFSPolicy()
    if normalized == "fixed_priority":
        return FixedPriorityPolicy()
    if normalized == "least_loaded":
        return LeastLoadedPolicy()
    if normalized == "work_stealing":
        return WorkStealingPolicy()
    if normalized == "heft":
        return HEFTPolicy()
    if normalized == "feedback_threshold":
        return FeedbackThresholdPolicy()
    if normalized == "plan6":
        return ProposedStateEvolutionPolicy(
            "plan6",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.18,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_stochastic":
        return ProposedStateEvolutionPolicy(
            "plan6_stochastic",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=True,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.18,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_nomigration":
        return ProposedStateEvolutionPolicy(
            "plan6_nomigration",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=False,
                feedback_enabled=True,
                energy_weight=0.18,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_nofeedback":
        return ProposedStateEvolutionPolicy(
            "plan6_nofeedback",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=False,
                energy_weight=0.18,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_nothermal":
        return ProposedStateEvolutionPolicy(
            "plan6_nothermal",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.18,
                thermal_weight=0.0,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=0.0,
            ),
        )
    if normalized == "plan6_noenergy":
        return ProposedStateEvolutionPolicy(
            "plan6_noenergy",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.0,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_deterministic":
        return ProposedStateEvolutionPolicy(
            "plan6_deterministic",
            PolicyConfig(
                eta=0.7,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.1,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.18,
                thermal_weight=0.24,
                queue_weight=0.20,
                exec_weight=0.24,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=8.0,
            ),
        )
    if normalized == "plan6_v2":
        return PredictiveLyapunovPolicy(
            "plan6_v2",
            PolicyConfig(
                eta=0.68,
                tau=0.15,
                migration_threshold=0.45,
                net_gain_threshold=0.22,
                probabilistic=False,
                migration_enabled=True,
                feedback_enabled=True,
                energy_weight=0.16,
                thermal_weight=0.10,
                queue_weight=0.18,
                exec_weight=0.18,
                migration_weight=0.10,
                bandwidth_weight=0.08,
                thermal_safety_margin=7.0,
                completion_weight=0.01,
                criticality_weight=0.22,
                starvation_weight=0.10,
                slack_weight=0.02,
                thermal_virtual_weight=0.55,
                bandwidth_virtual_weight=0.20,
                migration_virtual_weight=0.22,
                queue_forecast_weight=0.08,
                thermal_target_ratio=0.82,
                bandwidth_target=0.85,
                migration_budget=0.8,
                virtual_queue_decay=0.78,
            ),
        )
    raise KeyError(f"Unknown policy {name}")
