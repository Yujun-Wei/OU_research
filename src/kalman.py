"""Kalman filter for the AR(1) coefficients of a spread.

The state is the pair (a, b) in

    S_k = a S_{k-1} + b + ε_k

on pairs one minute apart inside the same session. The coefficients are a
random walk, so the transition matrix is the identity. The observation row is
[S_{k-1}, 1] and the measurement variance R is the OLS residual variance.
The filter is initialized at that OLS fit, with P0 equal to its coefficient
covariance.

a is clipped to [a_min, a_max] after each update. The row stored on a minute
is the state from before that minute's print, so the print is not used to
trade itself. Q is added only on a valid pair. A lunch gap or an overnight
gap carries the previous posterior forward and does not add process noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

A_MIN = 0.001
A_MAX = 0.999


@dataclass(frozen=True)
class AR1OLS:
    """In-sample AR(1) fit that anchors R, the initial state, and P0."""

    a: float
    b: float
    sigma_eps: float
    n_pairs: int
    n_bars: int
    p_aa: float
    p_ab: float
    p_bb: float

    @property
    def state(self) -> np.ndarray:
        return np.array([self.a, self.b], dtype=np.float64)

    @property
    def P0(self) -> np.ndarray:
        return np.array([[self.p_aa, self.p_ab], [self.p_ab, self.p_bb]], dtype=np.float64)

    @property
    def R(self) -> float:
        return self.sigma_eps * self.sigma_eps


@dataclass(frozen=True)
class KalmanPath:
    """Tradable parameters on each one-minute bar, known before that bar."""

    frame: pl.DataFrame
    n_updates: int
    n_clipped: int


def minute_bars(panel: pl.DataFrame, column: str) -> pl.DataFrame:
    """Last print of `column` in each clock minute. Empty minutes are left out."""
    if column not in panel.columns:
        raise ValueError(f"Panel has no column {column}")
    bars = (
        panel.sort("trade_time")
        .group_by(pl.col("trade_time").dt.truncate("1m").alias("trade_time"), maintain_order=True)
        .agg(pl.col(column).last())
        .drop_nulls([column])
        .sort("trade_time")
        .with_columns(pl.col("trade_time").dt.date().alias("session"))
    )
    if bars.height < 3:
        raise ValueError(f"{column}: need at least 3 one-minute bars, got {bars.height}")
    return bars


def fit_ar1_ols(minute: pl.DataFrame, column: str) -> AR1OLS:
    """OLS of S_k on S_{k-1}, and the coefficient covariance σ_ε² (X'X)⁻¹.

    σ_ε uses n − 2, the same residual degrees of freedom as `fit_ols`. Pairs
    that cross the lunch gap or the overnight gap are left out.
    """
    if column not in minute.columns:
        raise ValueError(f"Minute bars have no column {column}")
    lagged = (
        minute.sort("trade_time")
        .with_columns(
            pl.col(column).shift(1).over("session").alias("lag"),
            pl.col("trade_time").shift(1).over("session").alias("t_lag"),
        )
        .filter(
            pl.col("lag").is_not_null()
            & ((pl.col("trade_time") - pl.col("t_lag")).dt.total_minutes() == 1)
        )
    )
    y = lagged[column].to_numpy().astype(np.float64)
    x = lagged["lag"].to_numpy().astype(np.float64)
    n = int(y.size)
    if n < 3:
        raise ValueError(f"{column}: need at least 3 one-minute pairs, got {n}")
    dx = x - x.mean()
    var_x = float(dx @ dx)
    if var_x == 0.0:
        raise ValueError(f"{column}: lagged spread has zero variance")
    a = float((y - y.mean()) @ dx / var_x)
    b = float(y.mean() - a * x.mean())
    if not 0.0 < a < 1.0:
        raise ValueError(f"{column}: AR(1) coefficient a={a} is outside (0, 1)")
    resid = y - (a * x + b)
    sigma_eps = float(math.sqrt(float(resid @ resid) / (n - 2)))
    design = np.column_stack((x, np.ones(n, dtype=np.float64)))
    gram = design.T @ design
    cov = (sigma_eps * sigma_eps) * np.linalg.inv(gram)
    cov = 0.5 * (cov + cov.T)
    return AR1OLS(
        a=a,
        b=b,
        sigma_eps=sigma_eps,
        n_pairs=n,
        n_bars=minute.height,
        p_aa=float(cov[0, 0]),
        p_ab=float(cov[0, 1]),
        p_bb=float(cov[1, 1]),
    )


def kalman_filter(
    minute: pl.DataFrame,
    column: str,
    ols: AR1OLS,
    qa: float,
    qb: float,
    *,
    a_min: float = A_MIN,
    a_max: float = A_MAX,
) -> KalmanPath:
    """Run the random-walk filter and return the pre-update OU parameters.

    `ols` supplies the initial state, P0, and σ_ε. Those stay fixed for the
    run, which is how an in-sample fit is carried into later minutes. qa and
    qb are the diagonal entries of Q.
    """
    if column not in minute.columns:
        raise ValueError(f"Minute bars have no column {column}")
    if qa < 0.0 or qb < 0.0:
        raise ValueError(f"Process variances must be non-negative, got qa={qa} qb={qb}")
    if not 0.0 < a_min < a_max < 1.0:
        raise ValueError(f"Need 0 < a_min < a_max < 1, got {a_min}, {a_max}")
    if ols.sigma_eps <= 0.0:
        raise ValueError(f"sigma_eps must be positive, got {ols.sigma_eps}")

    bars = minute.sort("trade_time")
    values = bars[column].to_numpy().astype(np.float64)
    times = bars["trade_time"].to_numpy()
    sessions = bars["session"].to_numpy()
    n = int(values.size)
    if n < 3:
        raise ValueError(f"{column}: need at least 3 one-minute bars, got {n}")

    x = ols.state.copy()
    if x[0] < a_min:
        x[0] = a_min
    elif x[0] > a_max:
        x[0] = a_max
    P = ols.P0.copy()
    R = ols.R
    one_minute = np.timedelta64(60, "s")
    a_out = np.empty(n, dtype=np.float64)
    b_out = np.empty(n, dtype=np.float64)
    n_updates = 0
    n_clipped = 0

    for i in range(n):
        a_out[i] = x[0]
        b_out[i] = x[1]
        if i == 0:
            continue
        if sessions[i] != sessions[i - 1] or (times[i] - times[i - 1]) != one_minute:
            continue
        x, P, clipped = _update(x, P, float(values[i - 1]), float(values[i]), R, qa, qb, a_min, a_max)
        n_updates += 1
        n_clipped += int(clipped)

    theta = -np.log(a_out)
    mu = b_out / (1.0 - a_out)
    sigma = ols.sigma_eps * np.sqrt(2.0 * theta / (1.0 - a_out * a_out))
    sigma_eq = sigma / np.sqrt(2.0 * theta)
    frame = pl.DataFrame(
        {
            "trade_time": bars["trade_time"],
            "a": a_out,
            "b": b_out,
            "mu": mu,
            "theta_per_min": theta,
            "sigma": sigma,
            "sigma_eq": sigma_eq,
        }
    )
    return KalmanPath(frame=frame, n_updates=n_updates, n_clipped=n_clipped)


def _update(
    x: np.ndarray,
    P: np.ndarray,
    lag: float,
    observation: float,
    R: float,
    qa: float,
    qb: float,
    a_min: float,
    a_max: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Random-walk predict and Joseph-form update. Clips a after the update."""
    p00 = P[0, 0] + qa
    p01 = P[0, 1]
    p11 = P[1, 1] + qb
    hP0 = p00 * lag + p01
    hP1 = p01 * lag + p11
    scale = lag * hP0 + hP1 + R
    innovation = observation - (x[0] * lag + x[1])
    k0 = hP0 / scale
    k1 = hP1 / scale
    a_new = x[0] + k0 * innovation
    b_new = x[1] + k1 * innovation
    m00 = 1.0 - k0 * lag
    m01 = -k0
    m10 = -k1 * lag
    m11 = 1.0 - k1
    t00 = m00 * p00 + m01 * p01
    t01 = m00 * p01 + m01 * p11
    t10 = m10 * p00 + m11 * p01
    t11 = m10 * p01 + m11 * p11
    # (I − KH) P (I − KH)' + K R K'
    n00 = t00 * m00 + t01 * m01 + R * k0 * k0
    n01 = t00 * m10 + t01 * m11 + R * k0 * k1
    n11 = t10 * m10 + t11 * m11 + R * k1 * k1
    clipped = a_new < a_min or a_new > a_max
    if a_new < a_min:
        a_new = a_min
    elif a_new > a_max:
        a_new = a_max
    return (
        np.array([a_new, b_new], dtype=np.float64),
        np.array([[n00, n01], [n01, n11]], dtype=np.float64),
        clipped,
    )
