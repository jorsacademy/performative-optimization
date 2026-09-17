from performative_optimization.config import (
    EnvironmentConfig,
    ExperimentConfig,
)
from performative_optimization.experiment import run_benchmark, run_single_seed
from performative_optimization.environment import PerformativeDemandEnvironment


EXPECTED_METHODS = {
    "naive_static",
    "repeated_optimization",
    "performatively_stable",
    "plugin_performative",
    "known_response_oracle",
}


def test_run_schema_feasibility_and_oracle_regret() -> None:
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(sensitivity=0.3),
        deployment_rounds=3,
        samples_per_round=30,
        response_design_points=4,
        response_samples_per_point=25,
        seeds=(7,),
    )
    rows = run_single_seed(cfg, 7)
    assert {row.method for row in rows} == EXPECTED_METHODS
    assert all(row.feasible for row in rows)
    assert all(row.constraint_violation == 0.0 for row in rows)
    oracle = next(row for row in rows if row.method == "known_response_oracle")
    assert abs(oracle.regret_to_oracle) < 1e-12


def test_ood_reuses_nominal_response_model_without_recalibration() -> None:
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(sensitivity=0.2),
        ood_sensitivity=0.6,
        deployment_rounds=2,
        samples_per_round=25,
        response_design_points=4,
        response_samples_per_point=30,
        seeds=(13,),
    )
    nominal = PerformativeDemandEnvironment(cfg.environment)
    ood = PerformativeDemandEnvironment(
        EnvironmentConfig(
            base_mean=cfg.environment.base_mean,
            sensitivity=cfg.ood_sensitivity,
            sigma=cfg.environment.sigma,
            capacity_cost=cfg.environment.capacity_cost,
            shortage_cost=cfg.environment.shortage_cost,
            overage_cost=cfg.environment.overage_cost,
            min_capacity=cfg.environment.min_capacity,
            max_capacity=cfg.environment.max_capacity,
        )
    )
    test_rows = run_single_seed(cfg, 13, "test", nominal, nominal)
    ood_rows = run_single_seed(cfg, 13, "ood", ood, nominal)
    test_plugin = next(row for row in test_rows if row.method == "plugin_performative")
    ood_plugin = next(row for row in ood_rows if row.method == "plugin_performative")
    test_stable = next(row for row in test_rows if row.method == "performatively_stable")
    ood_stable = next(row for row in ood_rows if row.method == "performatively_stable")
    assert test_plugin.final_decision == ood_plugin.final_decision
    assert test_stable.final_decision == ood_stable.final_decision
    assert ood_plugin.performative_risk != test_plugin.performative_risk


def test_smoke_benchmark_contains_statistics_and_paired_differences() -> None:
    cfg = ExperimentConfig(
        deployment_rounds=2,
        samples_per_round=20,
        response_design_points=3,
        response_samples_per_point=20,
        seeds=(3, 5),
    )
    result = run_benchmark(cfg, include_ood=True)
    records = result["records"]
    summary = result["summary"]
    paired = result["paired_differences"]
    assert isinstance(records, list)
    assert isinstance(summary, dict)
    assert isinstance(paired, dict)
    assert {row["split"] for row in records} == {"test", "ood"}
    assert len(records) == 2 * 2 * len(EXPECTED_METHODS)
    assert "test:plugin_performative" in summary
    assert "test:plugin_performative-minus-naive_static" in paired
    assert "ood-minus-test:plugin_performative" in paired
