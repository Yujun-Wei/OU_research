"""Static OLS hedge of 511090 last on TL last, and the corrected spread.

The fit uses the first half of trading sessions. The same session cut is
applied at every bar size. The stored column `x_static` is the 1-minute hedge
applied to every row of the merged panel:

    x_static = P_ETF - beta * P_TL - alpha
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

TRAIN_FRAC = 0.5


@dataclass(frozen=True)
class HedgeFit:
    """Static OLS hedge of the ETF on TL."""

    frequency: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    n_train_sessions: int
    n_test_sessions: int
    n_train: int
    n_test: int
    alpha: float
    beta: float
    r_squared: float
    resid_std_train: float
    resid_mean_test: float
    resid_std_test: float


def resample_last(panel: pl.DataFrame, every: str) -> pl.DataFrame:
    """Last ETF and TL print in each clock bin. Empty bins are left out."""
    return (
        panel.sort("trade_time")
        .group_by(pl.col("trade_time").dt.truncate(every).alias("trade_time"), maintain_order=True)
        .agg(
            pl.col("etf_last").last(),
            pl.col("tl_last").last(),
        )
        .sort("trade_time")
    )


def session_cutoff(bars: pl.DataFrame, frac: float = TRAIN_FRAC) -> date:
    """Last in-sample session: the earliest `frac` of trading dates, rounded down."""
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0, 1), got {frac}")
    days = bars.select(pl.col("trade_time").dt.date().alias("session")).unique().sort("session")
    n_train = int(days.height * frac)
    if n_train < 1 or n_train >= days.height:
        raise ValueError(f"Need a split inside the sample, got {n_train} of {days.height} sessions")
    cutoff = days["session"][n_train - 1]
    if isinstance(cutoff, date):
        return cutoff
    return date.fromisoformat(str(cutoff))


def ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """beta = Cov(y, x) / Var(x), alpha = E[y] - beta E[x]."""
    if y.ndim != 1 or x.shape != y.shape:
        raise ValueError("y and x must be 1-d and the same length")
    if y.size < 3:
        raise ValueError(f"Need at least 3 observations, got {y.size}")
    dx = x - x.mean()
    var_x = float(dx @ dx)
    if var_x == 0.0:
        raise ValueError("TL price has zero variance on the training sample")
    beta = float((y - y.mean()) @ dx / var_x)
    alpha = float(y.mean() - beta * x.mean())
    return alpha, beta


def fit_hedge(bars: pl.DataFrame, train_end: date, frequency: str) -> HedgeFit:
    """Static hedge on sessions through `train_end`."""
    marked = bars.with_columns((pl.col("trade_time").dt.date() <= train_end).alias("in_sample"))
    train = marked.filter(pl.col("in_sample"))
    test = marked.filter(~pl.col("in_sample"))
    if train.is_empty() or test.is_empty():
        raise ValueError(f"{frequency}: training or test sample is empty")

    alpha, beta = ols_alpha_beta(train["etf_last"].to_numpy(), train["tl_last"].to_numpy())
    train_x = train["etf_last"].to_numpy() - beta * train["tl_last"].to_numpy() - alpha
    test_x = test["etf_last"].to_numpy() - beta * test["tl_last"].to_numpy() - alpha
    train_mean = float(train_x.mean())
    if abs(train_mean) > 1e-6:
        raise RuntimeError(f"{frequency}: in-sample spread mean is {train_mean}, expected 0")

    y = train["etf_last"].to_numpy()
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        raise ValueError(f"{frequency}: ETF price has zero variance on the training sample")
    r_squared = 1.0 - float(np.sum(train_x**2)) / ss_tot

    train_start = train["trade_time"].dt.date().min()
    test_start = test["trade_time"].dt.date().min()
    test_end = test["trade_time"].dt.date().max()
    fit = HedgeFit(
        frequency=frequency,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        n_train_sessions=train["trade_time"].dt.date().n_unique(),
        n_test_sessions=test["trade_time"].dt.date().n_unique(),
        n_train=train.height,
        n_test=test.height,
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        resid_std_train=float(train_x.std(ddof=1)),
        resid_mean_test=float(test_x.mean()),
        resid_std_test=float(test_x.std(ddof=1)),
    )
    return fit


def attach_static_hedge(panel: pl.DataFrame, alpha: float, beta: float) -> pl.DataFrame:
    """Add `x_static`. Replaces it if present."""
    existing = [name for name in ("spread", "x_static", "in_sample") if name in panel.columns]
    base = panel.drop(existing) if existing else panel
    return base.with_columns(
        (pl.col("etf_last") - beta * pl.col("tl_last") - alpha).alias("x_static"),
    )


def write_panel(panel: pl.DataFrame, path: Path) -> None:
    """Replace `path` with `panel`. A crash during the write leaves the old file."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        panel.write_parquet(tmp, compression="zstd")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
