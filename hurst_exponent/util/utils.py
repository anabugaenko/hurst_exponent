from numpy._typing import NDArray
import pandas as pd
import numpy as np
from numpy import ndarray
from typing import Optional, Union, Any, Callable, Tuple

from statsmodels.tsa.stattools import adfuller
from stochastic.processes.continuous import GeometricBrownianMotion

Series = NDArray[np.float64]


# ADF test
def augmented_dickey_fuller(sample: pd.Series, name: str, alpha: float = 0.05) -> None:
    # Convert list to pandas Series if necessary
    if isinstance(sample, list):
        sample = pd.Series(sample)

    # Replace inf and -inf with nan
    sample.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop nan values
    sample.dropna(inplace=True)

    print(name)
    adf, p, usedlag, nobs, cvs, aic = adfuller(sample)
    adf_results_string = "ADF: {:.2f}\np-value: {:.2f},\nN: {}, \ncritical values: {}"
    print(adf_results_string.format(adf, p, nobs, cvs))

    # Interpretation
    if p < alpha:
        print("We reject the null hypothesis at a significance level of {}.".format(alpha))
        print("The time series appears to be stationary.")
    else:
        print("We cannot reject the null hypothesis at a significance level of {}.".format(alpha))
        print("The time series appears to be non-stationary.")

    print("\n")


# Boostrap
def bootstrap(
    estimator: Callable[[Any], Tuple[float, Any]],
    reps: int,
    seed: Optional[int] = None,
) -> Series:
    """
    Bootstrap estimates using a provided estimator on Geometric Brownian Motion samples.

    Parameters
    ----------
    estimator : Callable[[Any], Tuple[float, Any]]
        Function that returns an estimate based on a sample as its first tuple element.

    reps : int
        Number of bootstrap repetitions.

    seed : int, optional
        Seed for reproducibility. Defaults to None.

    Returns
    -------
    np.array
        Array of bootstrapped estimates.

    """
    if seed:
        np.random.seed(seed)
    gbm = GeometricBrownianMotion(volatility=0.00002)
    return np.array([estimator(gbm.sample(2048))[0] for _ in range(reps)])


# Helper functions


def _get_sums_of_chunks(series: Series, lag_size: int) -> Series:
    """
    Reshapes a series into chunks of size N and sums each chunk.

    Parameters
    ----------
    series : np.array
        The time series to process
    lag_size : int
        Lag size

    Returns
    -------
    np.array
        Summed values of each chunk
    """
    reshaped_series = series[: len(series) // lag_size * lag_size].reshape(-1, lag_size)
    return np.sum(reshaped_series, axis=1)


def std_of_sums(ts: Series, lag_size: int) -> Union[ndarray, Any]:
    """
    Computes the standard deviation of sums of time series lags of size lag_size.

    :math: Sigma_{sqrt {|X(t+τ)−X(t)|^2}}

    Parameters
    ----------
    ts : np.array
        Time series data
    lag_size : int
        The size of each lag of the time series

    Returns
    -------
    std : float
        The standard deviation of the sums
    """
    if lag_size == 0:
        return np.nan

    sums = _get_sums_of_chunks(ts, lag_size)
    return np.std(sums)


def _calculate_diffs(ts: Series, lag: int) -> Series:
    """
    Calculate detrended differences at specified lag steps in the time series.

    x(t + lag) - x(t)

    Parameters
    ----------
    ts : np.array
        The time series data
    lag : int
        The step size to compute the differences

    Returns
    -------
    diffs : np.ndarray
        Detrended differences of the time series at specified lags
    """
    return ts[:-lag] - ts[lag:]


def structure_function(ts: Series, moment: int, lag: int) -> Union[ndarray, Any]:
    """
    Calculate the structure function for a given moment and lag,
    defined as the mean of the absolute differences to the power
    of the specified moment.

    :math: S_q(lag) = < | x(t + lag) - x(t) |^q >_t/(T −τ +1)

    Parameters
    ----------
    ts : np.array
        The time series data
    moment : int
        The moment for which the structure function is to be calculated
    lag : int
        The lag at which the structure function is to be calculated

    Returns
    -------
    S_q_tau : float
        The calculated structure function for the specified moment and lag.
        If the differences array is empty, it returns np.nan
    """
    diffs = np.abs(_calculate_diffs(ts, lag))
    ts_abs_moment = np.abs(ts[:-lag]) ** moment
    if diffs.size != 0 and np.any(ts_abs_moment):
        return np.mean(diffs**moment)
    else:
        return np.nan


def interpret_hurst(H: float) -> str:
    """
    Interpretation of Hurst Exponent, which represents a measure of the
    long-term memory. Hurst parameter :math:`H`. When `H = 1/2`,
    the process is a standard Brownian motion. `H > 1/2`, the
    increments are positively correlated. For `H < 1/2`, the
    increments are negatively correlated.

    Parameters
    ----------
    H : float
        Hurst Exponent.

    Returns
    -------
    str
        Interpretation of Hurst Exponent.
    """
    if not 0 <= H <= 1:
        # FIXME: The Generalized Hurst for white noise is dimension dependent, and for 1D and 2D it is H_{q}^{1D}={\frac {1}{2}},\quad H_{q}^{2D}=-1.
        raise ValueError("Hurst exponent not in a valid range (0,1).")
    if np.isclose(H, 0.5):
        return "Diffusive: series is close to a Geometric or Brownian random walk"
    if H < 0.5:
        return "Sub-diffusive: series demonstrates anti-persistent behaviour"
    if H > 0.5:
        return "Super-diffusive: series demonstrates persistent long-range dependence"
    return "Invalid Hurst Exponent"
