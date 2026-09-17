import math

from performative_optimization.audit import audit_decision
from performative_optimization.config import EnvironmentConfig
from performative_optimization.environment import PerformativeDemandEnvironment
from performative_optimization.newsvendor import performative_risk


def test_feasibility_audit_rejects_out_of_bounds() -> None:
    env = PerformativeDemandEnvironment(EnvironmentConfig(max_capacity=100.0))
    audit = audit_decision(120.0, env)
    assert not audit.feasible
    assert audit.upper_bound_violation == 20.0
    assert math.isinf(audit.objective_recomputed)


def test_feasibility_audit_recomputes_objective() -> None:
    env = PerformativeDemandEnvironment(EnvironmentConfig())
    decision = 100.0
    audit = audit_decision(decision, env)
    assert audit.feasible
    assert audit.objective_recomputed == performative_risk(decision, env)


def test_feasibility_audit_rejects_nonfinite_decision() -> None:
    env = PerformativeDemandEnvironment(EnvironmentConfig())
    audit = audit_decision(float("nan"), env)
    assert not audit.feasible
    assert math.isinf(audit.lower_bound_violation)
    assert math.isinf(audit.upper_bound_violation)
