from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RESOURCE_TYPES = ("cpu", "gpu", "npu")


@dataclass(slots=True)
class ResourceSpec:
    resource_id: str
    resource_type: str
    speed: float
    power_base: float
    power_dynamic: float
    heat_gain: float
    cooling_rate: float
    bandwidth_capacity: float
    thermal_limit: float = 95.0
    ambient_temp: float = 40.0


@dataclass(slots=True)
class ResourceState:
    spec: ResourceSpec
    task_id: str | None = None
    temperature: float = 40.0
    busy_steps: int = 0
    migration_delay_remaining: int = 0


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    dag_id: str
    arrival_time: int
    priority: int
    latency_class: str
    task_type: str
    size: float
    bandwidth_demand: float
    checkpoint_interval: float
    migration_penalty: int
    service_time: dict[str, int]
    power_draw: dict[str, float]
    deps: list[str]


@dataclass(slots=True)
class TaskState:
    spec: TaskSpec
    status: str = "pending"
    remaining_fraction: float = 1.0
    current_resource_id: str | None = None
    start_time: int | None = None
    completion_time: int | None = None
    last_checkpoint_fraction: float = 0.0
    cooldown_remaining: int = 0
    migrations: int = 0
    progress_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completed_fraction(self) -> float:
        return max(0.0, 1.0 - self.remaining_fraction)

    @property
    def ready_for_checkpoint(self) -> bool:
        interval = self.spec.checkpoint_interval
        if interval <= 0:
            return False
        return self.completed_fraction >= self.last_checkpoint_fraction + interval

    def remaining_time_on(self, resource_type: str) -> int:
        return max(1, int(round(self.spec.service_time[resource_type] * self.remaining_fraction)))


@dataclass(slots=True)
class PolicyConfig:
    eta: float = 0.55
    tau: float = 0.35
    migration_threshold: float = 0.55
    net_gain_threshold: float = 0.15
    cooldown_steps: int = 3
    probabilistic: bool = True
    migration_enabled: bool = True
    feedback_enabled: bool = True
    energy_weight: float = 0.18
    thermal_weight: float = 0.24
    queue_weight: float = 0.20
    exec_weight: float = 0.22
    migration_weight: float = 0.12
    bandwidth_weight: float = 0.08
    thermal_safety_margin: float = 8.0
    completion_weight: float = 0.0
    criticality_weight: float = 0.0
    starvation_weight: float = 0.0
    slack_weight: float = 0.0
    thermal_virtual_weight: float = 0.0
    bandwidth_virtual_weight: float = 0.0
    migration_virtual_weight: float = 0.0
    queue_forecast_weight: float = 0.0
    thermal_target_ratio: float = 0.78
    bandwidth_target: float = 0.75
    migration_budget: float = 1.0
    virtual_queue_decay: float = 0.85


@dataclass(slots=True)
class ExperimentPaths:
    package_root: str
    data_dir: str
    results_dir: str
    logs_dir: str
