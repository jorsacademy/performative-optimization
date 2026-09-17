from __future__ import annotations

import math
from dataclasses import dataclass

from .environment import PerformativeDemandEnvironment
from .newsvendor import performative_risk


@dataclass(frozen=True)
class AuditResult:
    feasible: bool
    lower_bound_violation: float
    upper_bound_violation: float
    objective_recomputed: float


def audit_decision(decision: float, env: PerformativeDemandEnvironment) -> AuditResult:
    cfg = env.config
    if not math.isfinite(decision):
        return AuditResult(False, math.inf, math.inf, math.inf)
    lower = max(0.0, cfg.min_capacity - decision)
    upper = max(0.0, decision - cfg.max_capacity)
    feasible = lower <= 1e-12 and upper <= 1e-12
    objective = performative_risk(decision, env) if feasible else math.inf
    return AuditResult(feasible, lower, upper, objective)
