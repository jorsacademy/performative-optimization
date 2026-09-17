from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import EnvironmentConfig


@dataclass(frozen=True)
class PerformativeDemandEnvironment:
    config: EnvironmentConfig

    def mean(self, decision: float) -> float:
        return self.config.base_mean + self.config.sensitivity * decision

    def sample(self, decision: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
        if n <= 0:
            raise ValueError("n must be positive")
        return rng.normal(self.mean(decision), self.config.sigma, size=n).astype(float)

    def distribution_shift(self, decision_a: float, decision_b: float) -> float:
        """1-Wasserstein distance for equal-variance Normal distributions."""
        return abs(self.mean(decision_b) - self.mean(decision_a))
