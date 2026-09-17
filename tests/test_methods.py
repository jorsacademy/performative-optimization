import numpy as np

from performative_optimization.config import EnvironmentConfig
from performative_optimization.environment import PerformativeDemandEnvironment
from performative_optimization.methods import repeated_optimization, stable_from_response
from performative_optimization.newsvendor import static_optimum
from performative_optimization.response_model import LinearResponseModel


def test_stable_method_satisfies_fixed_point() -> None:
    cfg = EnvironmentConfig(sensitivity=0.25)
    env = PerformativeDemandEnvironment(cfg)
    model = LinearResponseModel(cfg.base_mean, cfg.sensitivity)
    trace = stable_from_response(model, env)
    q = trace.final_decision
    assert abs(static_optimum(model.predict_mean(q), cfg) - q) < 1e-8
    assert trace.fixed_point_evaluations > 0


def test_repeated_optimization_is_seed_deterministic_and_uses_budget() -> None:
    env = PerformativeDemandEnvironment(EnvironmentConfig())
    initial = env.sample(70.0, 50, np.random.default_rng(999))
    a = repeated_optimization(env, 70.0, initial, 5, 50, np.random.default_rng(123))
    b = repeated_optimization(env, 70.0, initial, 5, 50, np.random.default_rng(123))
    assert a.decisions == b.decisions
    assert a.deployments == 5
    assert a.demand_samples == 250
    assert a.optimization_calls == 5
    assert len(a.decisions) == 6
