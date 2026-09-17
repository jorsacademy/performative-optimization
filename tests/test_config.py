import pytest

from performative_optimization.config import EnvironmentConfig, ExperimentConfig


def test_config_rejects_invalid_sigma() -> None:
    with pytest.raises(ValueError):
        EnvironmentConfig(sigma=0.0)


def test_config_rejects_infeasible_initial_decision() -> None:
    env = EnvironmentConfig(max_capacity=100.0)
    with pytest.raises(ValueError):
        ExperimentConfig(environment=env, initial_decision=120.0)
