"""OLS estimates of the mean-reversion parameters of a stored spread.

`fit_ols` is one window. `trailing_schedule` refits every 180 minutes on a
trailing five sessions; that path is what the trailing backtest trades.
The regression is on pairs one minute apart inside the same session, so the
lunch gap and the overnight gap are left out:

    X_t = a X_{t-1} + b + ε_t

With Δt = 1 minute,

    θ = -ln(a),  μ = b / (1 - a),  σ = σ_ε sqrt(2θ / (1 - a²))

σ_ε is the residual standard error. The half-life is ln(2) / θ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

REFRESH_MINUTES = 180
LOOKBACK_SESSIONS = 5


@dataclass(frozen=True)
class OLSFit:
    column: str
    start: date
    end: date
    n_sessions: int
    n_bars: int
    n_pairs: int
    a: float
    b: float
    theta_per_min: float
    mu: float
    sigma: float
    sigma_eps: float
    sigma_eq: float
    half_life_min: float


def fit_ols(panel: pl.DataFrame, column: str) -> OLSFit:
    """OLS estimates of θ, μ, and σ for `column` on this panel."""
    if column not in panel.columns:
        raise ValueError(f"Panel has no column {column}")
    bars = (
        panel.sort("trade_time")
        .group_by(pl.col("trade_time").dt.truncate("1m").alias("trade_time"), maintain_order=True)
        .agg(pl.col(column).last())
        .sort("trade_time")
        .with_columns(pl.col("trade_time").dt.date().alias("session"))
    )
    lagged = (
        bars.with_columns(
            pl.col(column).shift(1).over("session").alias("lag"),
            pl.col("trade_time").shift(1).over("session").alias("t_lag"),
        )
        .filter(
            pl.col("lag").is_not_null()
            & ((pl.col("trade_time") - pl.col("t_lag")).dt.total_minutes() == 1)
        )
    )
    y = lagged[column].to_numpy()
    x = lagged["lag"].to_numpy()
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
    theta = float(-math.log(a))
    mu = float(b / (1.0 - a))
    sigma = float(sigma_eps * math.sqrt(2.0 * theta / (1.0 - a * a)))
    start = bars["session"].min()
    end = bars["session"].max()
    if not isinstance(start, date):
        start = date.fromisoformat(str(start))
    if not isinstance(end, date):
        end = date.fromisoformat(str(end))
    return OLSFit(
        column=column,
        start=start,
        end=end,
        n_sessions=bars["session"].n_unique(),
        n_bars=bars.height,
        n_pairs=n,
        a=a,
        b=b,
        theta_per_min=theta,
        mu=mu,
        sigma=sigma,
        sigma_eps=sigma_eps,
        sigma_eq=float(sigma / math.sqrt(2.0 * theta)),
        half_life_min=float(math.log(2.0) / theta),
    )


def trailing_schedule(
    panel: pl.DataFrame,
    column: str,
    train_end: date,
    *,
    refresh_minutes: int = REFRESH_MINUTES,
    lookback_sessions: int = LOOKBACK_SESSIONS,
) -> pl.DataFrame:
    """OU parameters on a trailing window, refreshed through the test half.

    This is the 180-minute refresh of θ, μ, and σ. It is separate from the
    20-session rolling hedge in `rolling_hedge`.

    Bars are 1-minute lasts of `column`. The lookback is `lookback_sessions`
    times the session length (five sessions of 240 minutes is 1,200 bars).
    The first window ends on the last in-sample minute. The next windows end
    every `refresh_minutes` completed minutes after that. `known_at` is one
    minute after the last bar in the window, so that bar is not traded on
    the fit that includes it.

    A window whose AR(1) coefficient is outside (0, 1) keeps the previous
    fit. The first window has to be valid.
    """
    if column not in panel.columns:
        raise ValueError(f"Panel has no column {column}")
    if refresh_minutes < 1:
        raise ValueError(f"refresh_minutes must be positive, got {refresh_minutes}")
    if lookback_sessions < 1:
        raise ValueError(f"lookback_sessions must be positive, got {lookback_sessions}")

    minute = (
        panel.sort("trade_time")
        .group_by(pl.col("trade_time").dt.truncate("1m").alias("trade_time"), maintain_order=True)
        .agg(pl.col(column).last())
        .drop_nulls([column])
        .sort("trade_time")
        .with_columns(pl.col("trade_time").dt.date().alias("session"))
    )
    if minute.height < 3:
        raise ValueError(f"{column}: need at least 3 one-minute bars, got {minute.height}")
    day_len = int(minute.group_by("session").len()["len"].median())
    lookback_bars = lookback_sessions * day_len
    sessions = minute["session"].to_list()
    if not any(session <= train_end for session in sessions):
        raise ValueError(f"{column}: no in-sample minutes through {train_end}")
    idx_cut = max(i for i, session in enumerate(sessions) if session <= train_end)
    if idx_cut >= minute.height - 1:
        raise ValueError(f"{column}: no out-of-sample minutes after {train_end}")
    if idx_cut + 1 < lookback_bars:
        raise ValueError(
            f"{column}: in-sample history {idx_cut + 1} bars is shorter than the {lookback_bars}-bar lookback"
        )

    times = minute["trade_time"].to_list()
    values = minute[column].to_list()
    previous: OLSFit | None = None
    rows: list[dict] = []
    for end in range(idx_cut, minute.height, refresh_minutes):
        start = max(0, end - lookback_bars + 1)
        window = pl.DataFrame(
            {"trade_time": times[start : end + 1], column: values[start : end + 1]}
        ).with_columns(pl.col("trade_time").cast(pl.Datetime("us")))
        try:
            fitted = fit_ols(window, column)
        except ValueError:
            fitted = None
        carried = fitted is None
        if fitted is None:
            if previous is None:
                raise ValueError(f"{column}: OLS fit at the train cutoff is invalid")
            fitted = previous
        else:
            previous = fitted
        rows.append(
            {
                "asof": times[end],
                "mu": fitted.mu,
                "sigma": fitted.sigma,
                "theta_per_min": fitted.theta_per_min,
                "sigma_eq": fitted.sigma_eq,
                "n_pairs": fitted.n_pairs,
                "carried_forward": carried,
            }
        )
    return pl.DataFrame(rows).with_columns(
        (pl.col("asof") + pl.duration(minutes=1)).cast(pl.Datetime("us")).alias("known_at"),
        pl.lit(lookback_bars).alias("lookback_bars"),
        pl.lit(refresh_minutes).alias("refresh_minutes"),
    )


def attach_ols(panel: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Join the latest OU fit whose `known_at` is at or before each bar."""
    carried = ("mu", "sigma", "theta_per_min", "sigma_eq")
    drop = [name for name in carried if name in panel.columns]
    base = panel.drop(drop) if drop else panel
    fitted = schedule.select(
        pl.col("known_at").cast(pl.Datetime("us")).alias("trade_time"),
        "mu",
        "sigma",
        "theta_per_min",
        "sigma_eq",
    ).sort("trade_time")
    return (
        base.with_columns(pl.col("trade_time").cast(pl.Datetime("us")))
        .sort("trade_time")
        .join_asof(fitted, on="trade_time", strategy="backward")
    )
