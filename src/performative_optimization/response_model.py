from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LinearResponseModel:
    intercept: float
    slope: float

    def predict_mean(self, decision: float) -> float:
        return self.intercept + self.slope * decision


def fit_linear_response(
    decisions: NDArray[np.float64],
    sample_means: NDArray[np.float64],
) -> LinearResponseModel:
    if decisions.ndim != 1 or sample_means.ndim != 1 or len(decisions) != len(sample_means):
        raise ValueError("decisions and sample_means must be aligned one-dimensional arrays")
    if len(decisions) < 2 or np.allclose(decisions, decisions[0]):
        raise ValueError("at least two distinct decision points are required")
    x = np.column_stack([np.ones_like(decisions), decisions])
    beta, *_ = np.linalg.lstsq(x, sample_means, rcond=None)
    return LinearResponseModel(intercept=float(beta[0]), slope=float(beta[1]))
