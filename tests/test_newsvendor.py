from performative_optimization.config import EnvironmentConfig
from performative_optimization.environment import PerformativeDemandEnvironment
from performative_optimization.newsvendor import (
    bounded_numerical_oracle,
    grid_oracle,
    performative_oracle,
    performative_risk,
    static_optimum,
    static_risk,
)


def test_static_optimum_beats_nearby_points() -> None:
    cfg = EnvironmentConfig()
    mu = 95.0
    q = static_optimum(mu, cfg)
    assert static_risk(q, mu, cfg) <= static_risk(q - 0.1, mu, cfg)
    assert static_risk(q, mu, cfg) <= static_risk(q + 0.1, mu, cfg)


def test_analytic_oracle_agrees_with_two_independent_checks() -> None:
    env = PerformativeDemandEnvironment(EnvironmentConfig())
    q_exact, r_exact = performative_oracle(env)
    q_numeric, r_numeric = bounded_numerical_oracle(env)
    q_grid, r_grid = grid_oracle(env, points=20_001)

    assert abs(q_exact - q_numeric) < 1e-5
    assert abs(r_exact - r_numeric) < 1e-9
    assert abs(q_exact - q_grid) < 0.05
    assert abs(r_exact - r_grid) < 1e-3
    assert performative_risk(q_exact, env) == r_exact


def test_oracle_handles_boundary_solution() -> None:
    cfg = EnvironmentConfig(
        base_mean=15.0,
        sensitivity=0.0,
        capacity_cost=10.0,
        shortage_cost=2.0,
        overage_cost=1.0,
        max_capacity=50.0,
    )
    env = PerformativeDemandEnvironment(cfg)
    q, _ = performative_oracle(env)
    assert q == cfg.min_capacity
