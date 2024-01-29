""" Tests for hurst estimators """
import functools
from collections.abc import Callable
from functools import lru_cache
from typing import Iterable, Tuple, Any
from numpy.typing import NDArray


import numpy as np
import pytest
from powerlaw_function import Fit
from typing_extensions import List


from hurst_exponent import standard_hurst, generalized_hurst


Estimator = Callable[[NDArray[np.float64]], Fit]
Estimators = List[Tuple[str, Estimator]]


def all_estimators_for(
    base_name: str,
    base_estimator: Callable,
    max_lags: Iterable[float] = (256, 512, 1024),
    fitting_methods: Iterable[str] = ("MLE", "Least_squares"),
) -> Estimators:
    """Create estimators for all hyperparameters of the specified base estimator
    in the form [(estimator_name, estimator_fn]), ...]"""
    return [
        (
            f"{base_name} [{method}-{max_lag}]",
            functools.partial(base_estimator, fitting_method=method, max_lag=max_lag),
        )
        for max_lag in max_lags
        for method in fitting_methods
    ]


def all_estimators() -> Estimators:
    """Create estimators used for all tests in the form [(estimator_name, estimator_fn]), ...]"""
    return all_estimators_for("generalised", generalized_hurst) + all_estimators_for("standard", standard_hurst)


def gbm(length: int, volatility: float, initial_value: float = 1.0) -> NDArray[np.float64]:
    """Simulate Geometric Brownian Motion (GBM) with the given parameters."""
    return initial_value * np.exp(np.cumsum(np.random.normal(size=length, scale=volatility)))


@lru_cache
def bootstrap(
    estimator: Estimator,
    reps: int = 1000,
    seed: int = 42,
    length: int = 2048,
    volatility: float = 0.00002,
) -> NDArray[np.float64]:
    """
    Perform a bootstrap where we compute the specified Hurst estimator for a
    white noise process (H=0.5) with the specified sample size and volatility.
    We compute each estimate iid. for the same data generation process,
    and fix the seed once ahead of generating the iid. samples.
    """
    np.random.seed(seed)
    return np.array([estimator(gbm(length=length, volatility=volatility))[0] for _repetition in range(reps)])


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_unbiased_estimator(_estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives an unbiased estimate of H=0.5 for white noise."""
    point_estimates = bootstrap(estimator)
    assert np.isclose(np.mean(point_estimates), 0.5, rtol=1e-2)


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_within_limits(_estimator_name: str, estimator: Estimator):
    """Check whether the estimator gives valid point estimates within interval (0, 1)."""
    point_estimates = bootstrap(estimator)
    assert np.min(point_estimates) >= 0.0
    assert np.max(point_estimates) <= 1.0


@pytest.mark.parametrize(["_estimator_name", "estimator"], all_estimators())
def test_confidence_interval(_estimator_name: str, estimator: Estimator):
    """
    Check whether the estimator gives a 95% confidence interval whose bounds are within
    the worst-case reported in the literature.  See Barunuk (2010) and Weron (2002).

    Barunik, Jozef, and Ladislav Kristoufek. "On Hurst exponent estimation under
    heavy-tailed distributions." Physica A: statistical mechanics and its
    applications 389.18 (2010): 3844-3855.

    Weron, Rafał. "Estimating long-range dependence: finite sample properties and
    confidence intervals." Physica A: Statistical Mechanics and its Applications
    312.1-2 (2002): 285-299.
    """
    point_estimates = bootstrap(estimator)
    lower_ci = np.percentile(point_estimates, 2.5)
    upper_ci = np.percentile(point_estimates, 97.5)
    assert upper_ci < 0.75
    assert lower_ci > 0.25
