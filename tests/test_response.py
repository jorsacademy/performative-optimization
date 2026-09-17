import numpy as np
import pytest

from performative_optimization.response_model import fit_linear_response


def test_response_fit_recovers_exact_line() -> None:
    q = np.asarray([0.0, 1.0, 2.0, 3.0])
    mu = 7.0 + 0.4 * q
    model = fit_linear_response(q, mu)
    assert abs(model.intercept - 7.0) < 1e-12
    assert abs(model.slope - 0.4) < 1e-12


def test_response_fit_rejects_degenerate_design() -> None:
    q = np.asarray([1.0, 1.0, 1.0])
    mu = np.asarray([2.0, 2.1, 1.9])
    with pytest.raises(ValueError):
        fit_linear_response(q, mu)
