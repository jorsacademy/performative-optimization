from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    base_mean: float = 80.0
    sensitivity: float = 0.35
    sigma: float = 12.0
    capacity_cost: float = 1.0
    shortage_cost: float = 5.0
    overage_cost: float = 1.0
    min_capacity: float = 0.0
    max_capacity: float = 180.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.shortage_cost < 0.0 or self.overage_cost < 0.0:
            raise ValueError("shortage_cost and overage_cost must be nonnegative")
        if self.shortage_cost + self.overage_cost <= 0.0:
            raise ValueError("shortage_cost + overage_cost must be positive")
        if self.min_capacity >= self.max_capacity:
            raise ValueError("min_capacity must be strictly smaller than max_capacity")


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig = EnvironmentConfig()
    initial_decision: float = 70.0
    deployment_rounds: int = 20
    samples_per_round: int = 250
    response_design_points: int = 8
    response_samples_per_point: int = 150
    seeds: tuple[int, ...] = (11, 29, 47, 83, 101, 131, 173, 211, 257, 307)
    ood_sensitivity: float = 0.55

    def __post_init__(self) -> None:
        if not (
            self.environment.min_capacity <= self.initial_decision <= self.environment.max_capacity
        ):
            raise ValueError("initial_decision must be feasible")
        if self.deployment_rounds < 1:
            raise ValueError("deployment_rounds must be at least 1")
        if self.samples_per_round < 2:
            raise ValueError("samples_per_round must be at least 2")
        if self.response_design_points < 2:
            raise ValueError("response_design_points must be at least 2")
        if self.response_samples_per_point < 2:
            raise ValueError("response_samples_per_point must be at least 2")
        if not self.seeds:
            raise ValueError("at least one seed is required")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    env = EnvironmentConfig(**raw.get("environment", {}))
    rest = {key: value for key, value in raw.items() if key != "environment"}
    if "seeds" in rest:
        rest["seeds"] = tuple(rest["seeds"])
    return ExperimentConfig(environment=env, **rest)
