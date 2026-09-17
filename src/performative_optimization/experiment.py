from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import t

from .audit import audit_decision
from .config import ExperimentConfig
from .environment import PerformativeDemandEnvironment
from .methods import (
    MethodTrace,
    known_response_oracle,
    naive_static,
    plugin_performative,
    repeated_optimization,
    stable_from_response,
)
from .newsvendor import performative_risk, static_optimum, static_risk
from .response_model import LinearResponseModel, fit_linear_response


@dataclass(frozen=True)
class RunResult:
    seed: int
    split: str
    method: str
    final_decision: float
    performative_risk: float
    naive_static_risk: float
    regret_to_oracle: float
    oscillation: float
    stability_residual: float
    distribution_shift: float
    deployments: int
    demand_samples: int
    optimization_calls: int
    fixed_point_evaluations: int
    feasible: bool
    constraint_violation: float


def _fit_response_model(
    env: PerformativeDemandEnvironment,
    cfg: ExperimentConfig,
    seed: int,
) -> LinearResponseModel:
    rng = np.random.default_rng(seed + 10_000)
    lo, hi = env.config.min_capacity, env.config.max_capacity
    decisions = np.linspace(
        lo + 0.1 * (hi - lo),
        hi - 0.1 * (hi - lo),
        cfg.response_design_points,
    )
    sample_means = np.asarray(
        [
            np.mean(env.sample(float(q), cfg.response_samples_per_point, rng))
            for q in decisions
        ],
        dtype=float,
    )
    return fit_linear_response(decisions.astype(float), sample_means)


def _evaluate_trace(
    trace: MethodTrace,
    env: PerformativeDemandEnvironment,
    seed: int,
    split: str,
    initial_mu: float,
    oracle_risk: float,
    initial_decision: float,
) -> RunResult:
    q = trace.final_decision
    audit = audit_decision(q, env)
    risk = audit.objective_recomputed
    violation = audit.lower_bound_violation + audit.upper_bound_violation
    stable_best_response = static_optimum(env.mean(q), env.config) if audit.feasible else float("inf")
    stability_residual = abs(stable_best_response - q) if audit.feasible else float("inf")
    return RunResult(
        seed=seed,
        split=split,
        method=trace.name,
        final_decision=q,
        performative_risk=risk,
        naive_static_risk=static_risk(q, initial_mu, env.config),
        regret_to_oracle=risk - oracle_risk,
        oscillation=trace.oscillation,
        stability_residual=stability_residual,
        distribution_shift=env.distribution_shift(initial_decision, q),
        deployments=trace.deployments,
        demand_samples=trace.demand_samples,
        optimization_calls=trace.optimization_calls,
        fixed_point_evaluations=trace.fixed_point_evaluations,
        feasible=audit.feasible,
        constraint_violation=violation,
    )


def run_single_seed(
    cfg: ExperimentConfig,
    seed: int,
    split: str = "test",
    evaluation_env: PerformativeDemandEnvironment | None = None,
    calibration_env: PerformativeDemandEnvironment | None = None,
) -> list[RunResult]:
    eval_env = evaluation_env or PerformativeDemandEnvironment(cfg.environment)
    calib_env = calibration_env or eval_env

    initial_rng = np.random.default_rng(seed)
    initial_samples = eval_env.sample(cfg.initial_decision, cfg.samples_per_round, initial_rng)
    initial_mu = float(np.mean(initial_samples))

    response_model = _fit_response_model(calib_env, cfg, seed)
    calibration_deployments = cfg.response_design_points
    calibration_samples = cfg.response_design_points * cfg.response_samples_per_point

    oracle = known_response_oracle(eval_env)
    oracle_risk = performative_risk(oracle.final_decision, eval_env)

    traces = [
        naive_static(initial_samples, eval_env),
        repeated_optimization(
            eval_env,
            cfg.initial_decision,
            initial_samples,
            cfg.deployment_rounds,
            cfg.samples_per_round,
            np.random.default_rng(seed + 1_000),
        ),
        stable_from_response(
            response_model,
            eval_env,
            calibration_deployments,
            calibration_samples,
        ),
        plugin_performative(
            response_model,
            eval_env,
            calibration_deployments,
            calibration_samples,
        ),
        oracle,
    ]
    return [
        _evaluate_trace(
            trace,
            eval_env,
            seed,
            split,
            initial_mu,
            oracle_risk,
            cfg.initial_decision,
        )
        for trace in traces
    ]


def _stats(values: list[float]) -> dict[str, float]:
    value_mean = mean(values)
    value_std = stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        sem = value_std / float(np.sqrt(len(values)))
        half_width = float(t.ppf(0.975, df=len(values) - 1) * sem)
    else:
        half_width = 0.0
    return {
        "mean": value_mean,
        "std": value_std,
        "median": median(values),
        "p90": float(np.quantile(values, 0.9)),
        "ci95_low": value_mean - half_width,
        "ci95_high": value_mean + half_width,
    }


def _paired_differences(
    rows: list[RunResult],
    split: str,
    method: str,
    baseline: str,
) -> dict[str, float]:
    method_by_seed = {
        row.seed: row.performative_risk
        for row in rows
        if row.split == split and row.method == method
    }
    baseline_by_seed = {
        row.seed: row.performative_risk
        for row in rows
        if row.split == split and row.method == baseline
    }
    shared = sorted(method_by_seed.keys() & baseline_by_seed.keys())
    differences = [method_by_seed[seed] - baseline_by_seed[seed] for seed in shared]
    return _stats(differences)


def run_benchmark(cfg: ExperimentConfig, include_ood: bool = True) -> dict[str, object]:
    rows: list[RunResult] = []
    nominal_env = PerformativeDemandEnvironment(cfg.environment)

    for seed in cfg.seeds:
        rows.extend(
            run_single_seed(
                cfg,
                seed,
                split="test",
                evaluation_env=nominal_env,
                calibration_env=nominal_env,
            )
        )

    if include_ood:
        ood_config = replace(cfg.environment, sensitivity=cfg.ood_sensitivity)
        ood_env = PerformativeDemandEnvironment(ood_config)
        for seed in cfg.seeds:
            rows.extend(
                run_single_seed(
                    cfg,
                    seed,
                    split="ood",
                    evaluation_env=ood_env,
                    calibration_env=nominal_env,
                )
            )

    records = [asdict(row) for row in rows]
    summary: dict[str, dict[str, float]] = {}
    methods = sorted({row.method for row in rows})
    splits = sorted({row.split for row in rows})

    for split in splits:
        for method in methods:
            method_rows = [row for row in rows if row.split == split and row.method == method]
            if not method_rows:
                continue
            risk_stats = _stats([row.performative_risk for row in method_rows])
            summary[f"{split}:{method}"] = {
                **{f"risk_{key}": value for key, value in risk_stats.items()},
                "mean_regret_to_oracle": mean(row.regret_to_oracle for row in method_rows),
                "mean_oscillation": mean(row.oscillation for row in method_rows),
                "mean_stability_residual": mean(
                    row.stability_residual for row in method_rows
                ),
                "mean_distribution_shift": mean(
                    row.distribution_shift for row in method_rows
                ),
                "mean_deployments": mean(row.deployments for row in method_rows),
                "mean_demand_samples": mean(row.demand_samples for row in method_rows),
                "mean_optimization_calls": mean(
                    row.optimization_calls for row in method_rows
                ),
            }

    paired: dict[str, dict[str, float]] = {}
    for split in splits:
        for baseline in ("naive_static", "repeated_optimization"):
            for method in methods:
                if method == baseline:
                    continue
                key = f"{split}:{method}-minus-{baseline}"
                paired[key] = _paired_differences(rows, split, method, baseline)

    if include_ood:
        for method in methods:
            test_by_seed = {
                row.seed: row.performative_risk
                for row in rows
                if row.split == "test" and row.method == method
            }
            ood_by_seed = {
                row.seed: row.performative_risk
                for row in rows
                if row.split == "ood" and row.method == method
            }
            shared = sorted(test_by_seed.keys() & ood_by_seed.keys())
            deltas = [ood_by_seed[seed] - test_by_seed[seed] for seed in shared]
            paired[f"ood-minus-test:{method}"] = _stats(deltas)

    return {
        "metadata": {
            "paired_seed_design": True,
            "ood_protocol": (
                "Response models are calibrated on the nominal environment and evaluated "
                "without recalibration after sensitivity shifts to ood_sensitivity."
            ),
            "paired_difference_sign": "negative means the first method has lower risk",
        },
        "records": records,
        "summary": summary,
        "paired_differences": paired,
    }


def write_benchmark_json(result: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
