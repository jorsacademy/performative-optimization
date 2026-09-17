from __future__ import annotations

import math

from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .config import EnvironmentConfig
from .environment import PerformativeDemandEnvironment


def expected_positive_part(mu: float, sigma: float, threshold: float) -> float:
    z = (threshold - mu) / sigma
    return float(sigma * norm.pdf(z) + (mu - threshold) * (1.0 - norm.cdf(z)))


def expected_negative_part(mu: float, sigma: float, threshold: float) -> float:
    z = (threshold - mu) / sigma
    return float(sigma * norm.pdf(z) + (threshold - mu) * norm.cdf(z))


def static_risk(decision: float, mu: float, cfg: EnvironmentConfig) -> float:
    shortage = expected_positive_part(mu, cfg.sigma, decision)
    overage = expected_negative_part(mu, cfg.sigma, decision)
    return float(
        cfg.capacity_cost * decision
        + cfg.shortage_cost * shortage
        + cfg.overage_cost * overage
    )


def performative_risk(decision: float, env: PerformativeDemandEnvironment) -> float:
    return static_risk(decision, env.mean(decision), env.config)


def static_optimum(mu: float, cfg: EnvironmentConfig) -> float:
    denominator = cfg.shortage_cost + cfg.overage_cost
    fractile = (cfg.shortage_cost - cfg.capacity_cost) / denominator
    candidates = [cfg.min_capacity, cfg.max_capacity]
    if 0.0 < fractile < 1.0:
        interior = mu + cfg.sigma * float(norm.ppf(fractile))
        if cfg.min_capacity <= interior <= cfg.max_capacity:
            candidates.append(interior)
    return float(min(candidates, key=lambda q: static_risk(q, mu, cfg)))


def decision_dependent_risk(
    decision: float,
    intercept: float,
    slope: float,
    cfg: EnvironmentConfig,
) -> float:
    return static_risk(decision, intercept + slope * decision, cfg)


def decision_dependent_optimum(
    intercept: float,
    slope: float,
    cfg: EnvironmentConfig,
) -> tuple[float, float]:
    """Global optimum for Normal demand with an affine decision-dependent mean.

    For mu(q) = intercept + slope*q, the risk is convex because its second
    derivative is proportional to (1-slope)^2 times the Normal density.
    We therefore compare the feasible stationary point, when it exists, with
    the two interval endpoints. This is an analytical OR oracle for the
    synthetic benchmark, not a generic performative-optimization solver.
    """

    denominator = cfg.shortage_cost + cfg.overage_cost
    one_minus_slope = 1.0 - slope
    candidates = [cfg.min_capacity, cfg.max_capacity]

    if abs(one_minus_slope) > 1e-12:
        fractile = (
            cfg.shortage_cost - cfg.capacity_cost / one_minus_slope
        ) / denominator
        if 0.0 < fractile < 1.0:
            z = float(norm.ppf(fractile))
            interior = (intercept + cfg.sigma * z) / one_minus_slope
            if cfg.min_capacity <= interior <= cfg.max_capacity:
                candidates.append(interior)

    best_q = min(
        candidates,
        key=lambda q: decision_dependent_risk(q, intercept, slope, cfg),
    )
    return float(best_q), decision_dependent_risk(best_q, intercept, slope, cfg)


def performative_oracle(env: PerformativeDemandEnvironment) -> tuple[float, float]:
    return decision_dependent_optimum(
        env.config.base_mean,
        env.config.sensitivity,
        env.config,
    )


def bounded_numerical_oracle(env: PerformativeDemandEnvironment) -> tuple[float, float]:
    cfg = env.config
    result = minimize_scalar(
        lambda q: performative_risk(float(q), env),
        bounds=(cfg.min_capacity, cfg.max_capacity),
        method="bounded",
        options={"xatol": 1e-11},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("bounded numerical oracle failed")
    return float(result.x), float(result.fun)


def grid_oracle(env: PerformativeDemandEnvironment, points: int = 100_001) -> tuple[float, float]:
    if points < 3:
        raise ValueError("points must be at least 3")
    cfg = env.config
    step = (cfg.max_capacity - cfg.min_capacity) / (points - 1)
    best_q = cfg.min_capacity
    best_risk = performative_risk(best_q, env)
    for index in range(1, points):
        q = cfg.min_capacity + index * step
        risk = performative_risk(q, env)
        if risk < best_risk:
            best_q, best_risk = q, risk
    return float(best_q), float(best_risk)
