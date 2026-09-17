from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .environment import PerformativeDemandEnvironment
from .newsvendor import decision_dependent_optimum, performative_oracle, static_optimum
from .response_model import LinearResponseModel


@dataclass(frozen=True)
class MethodTrace:
    name: str
    decisions: tuple[float, ...]
    deployments: int
    demand_samples: int
    optimization_calls: int
    fixed_point_evaluations: int = 0

    @property
    def final_decision(self) -> float:
        return self.decisions[-1]

    @property
    def oscillation(self) -> float:
        if len(self.decisions) < 2:
            return 0.0
        diffs = np.abs(np.diff(np.asarray(self.decisions, dtype=float)))
        return float(np.mean(diffs))


def naive_static(
    initial_samples: NDArray[np.float64],
    env: PerformativeDemandEnvironment,
) -> MethodTrace:
    decision = static_optimum(float(np.mean(initial_samples)), env.config)
    return MethodTrace(
        "naive_static",
        (decision,),
        deployments=1,
        demand_samples=len(initial_samples),
        optimization_calls=1,
    )


def repeated_optimization(
    env: PerformativeDemandEnvironment,
    initial_decision: float,
    initial_samples: NDArray[np.float64],
    rounds: int,
    samples_per_round: int,
    rng: np.random.Generator,
) -> MethodTrace:
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    decisions = [float(initial_decision)]
    samples = initial_samples
    for round_index in range(rounds):
        current = static_optimum(float(np.mean(samples)), env.config)
        decisions.append(current)
        if round_index < rounds - 1:
            samples = env.sample(current, samples_per_round, rng)
    return MethodTrace(
        "repeated_optimization",
        tuple(decisions),
        deployments=rounds,
        demand_samples=rounds * samples_per_round,
        optimization_calls=rounds,
    )


def stable_from_response(
    model: LinearResponseModel,
    env: PerformativeDemandEnvironment,
    calibration_deployments: int = 0,
    calibration_samples: int = 0,
) -> MethodTrace:
    cfg = env.config
    evaluations = 0

    def fixed_point_residual(q: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return static_optimum(model.predict_mean(q), cfg) - q

    lo, hi = cfg.min_capacity, cfg.max_capacity
    flo, fhi = fixed_point_residual(lo), fixed_point_residual(hi)
    if abs(flo) <= 1e-12:
        q = lo
    elif abs(fhi) <= 1e-12:
        q = hi
    else:
        q = float(brentq(fixed_point_residual, lo, hi, xtol=1e-12, rtol=1e-12))
    residual = abs(static_optimum(model.predict_mean(q), cfg) - q)
    if residual > 1e-8:
        raise RuntimeError(f"fixed-point solve did not converge: residual={residual}")
    return MethodTrace(
        "performatively_stable",
        (q,),
        deployments=calibration_deployments,
        demand_samples=calibration_samples,
        optimization_calls=1,
        fixed_point_evaluations=evaluations,
    )


def plugin_performative(
    model: LinearResponseModel,
    env: PerformativeDemandEnvironment,
    calibration_deployments: int = 0,
    calibration_samples: int = 0,
) -> MethodTrace:
    q, _ = decision_dependent_optimum(model.intercept, model.slope, env.config)
    return MethodTrace(
        "plugin_performative",
        (q,),
        deployments=calibration_deployments,
        demand_samples=calibration_samples,
        optimization_calls=1,
    )


def known_response_oracle(env: PerformativeDemandEnvironment) -> MethodTrace:
    q, _ = performative_oracle(env)
    return MethodTrace(
        "known_response_oracle",
        (q,),
        deployments=0,
        demand_samples=0,
        optimization_calls=1,
    )
