"""The spread book.

A short spread sells the ETF and buys TL when z is above the entry band, and
buys the ETF and sells TL when z is below the negative band. The position is
flattened when |z| is back inside the exit band. Size is round(|z|) TL lots,
at least one and at most eight, cut so the two notionals stay inside the
capital cap. Both legs fill on the current bar's last.

    z = (spread - μ) / σ_eq

`simulate` is the book. `arrays` turns a panel into the arrays it takes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl

MAX_CAPITAL = 10_000_000.0
MAX_LOTS = 8
TL_MULT = 10_000.0
Z_ENTER = 2.0
Z_EXIT = 0.5
# ETF commission: 0.5 bp of notional, each side.
ETF_COMMISSION = 0.00005
# 3 yuan per TL lot at a notional of 1.2 million.
TL_COMMISSION = 0.0000025
SIDES = ("short", "long", "symmetrical")


def arrays(panel: pl.DataFrame, spread: str, beta: float | str) -> dict[str, np.ndarray]:
    """Tick arrays for `simulate`. `beta` is a constant or a column name."""
    if spread not in panel.columns:
        raise ValueError(f"Panel has no column {spread}")
    hedge = pl.lit(float(beta)) if isinstance(beta, (int, float)) else pl.col(beta)
    optional = [name for name in ("mu", "sigma_eq") if name in panel.columns]
    frame = (
        panel.select(
            "trade_time",
            "etf_last",
            "tl_last",
            pl.col(spread).alias("spread"),
            hedge.alias("beta"),
            *[pl.col(name) for name in optional],
        )
        .drop_nulls(["spread", "beta", "etf_last", "tl_last", *optional])
        .sort("trade_time")
    )
    out = {
        "etf": frame["etf_last"].to_numpy().astype(np.float64),
        "tl": frame["tl_last"].to_numpy().astype(np.float64),
        "spread": frame["spread"].to_numpy().astype(np.float64),
        "beta": frame["beta"].to_numpy().astype(np.float64),
        "times": frame["trade_time"].to_numpy(),
    }
    for name in optional:
        out[name] = frame[name].to_numpy().astype(np.float64)
    return out


def print_metrics(label: str, metrics: dict) -> None:
    """One result line, shared by both backtests."""

    def pct(value: float) -> str:
        return "n/a" if not math.isfinite(value) else f"{value:.4%}"

    sharpe = metrics["sharpe_ratio"]
    sharpe_text = "n/a" if not math.isfinite(sharpe) else f"{sharpe:.3f}"
    print(
        f"  {label}: "
        f"absolute {pct(metrics['absolute_return'])}, "
        f"annualized {pct(metrics['annualized_return'])}, "
        f"sharpe {sharpe_text}, "
        f"max drawdown {pct(metrics['max_drawdown'])}, "
        f"trades {metrics['trades']}",
        flush=True,
    )


def simulate(
    etf: np.ndarray,
    tl: np.ndarray,
    spread: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    sigma_eq: np.ndarray,
    times: np.ndarray,
    *,
    sides: str = "symmetrical",
    z_enter: float | np.ndarray = Z_ENTER,
    z_exit: float | np.ndarray = Z_EXIT,
    max_capital: float = MAX_CAPITAL,
    max_lots: int = MAX_LOTS,
    etf_commission: float = ETF_COMMISSION,
    tl_commission: float = TL_COMMISSION,
    tl_mult: float = TL_MULT,
    start_value: float = MAX_CAPITAL,
    record: bool = False,
) -> dict:
    """One pass over the bars. `times` is the bar clock, used for the daily Sharpe.

    `record` keeps the z-score and the signed TL lots after each bar. A book
    still open on the last bar is flattened there, so the last lot is zero.
    """
    if sides not in SIDES:
        raise ValueError(f"sides must be one of {SIDES}, got {sides!r}")
    n = int(etf.shape[0])
    if any(arr.shape[0] != n for arr in (tl, spread, beta, mu, sigma_eq, times)):
        raise ValueError("etf, tl, spread, beta, mu, sigma_eq, and times must share a length")
    z_enter = _thresholds(z_enter, n, "z_enter")
    z_exit = _thresholds(z_exit, n, "z_exit")
    if np.any(z_exit > z_enter):
        raise ValueError("Need z_exit <= z_enter on every bar")
    allow_short = sides in {"short", "symmetrical"}
    allow_long = sides in {"long", "symmetrical"}
    equity = np.empty(n, dtype=np.float64)
    value = float(start_value)
    lots = 0
    shares = 0
    prev_etf = 0.0
    prev_tl = 0.0
    seen = False
    n_trades = 0
    z_path = np.full(n, np.nan) if record else None
    lots_path = np.zeros(n, dtype=np.int32) if record else None

    for i in range(n):
        etf_i = float(etf[i])
        tl_i = float(tl[i])
        if lots != 0 and seen:
            value += shares * (etf_i - prev_etf) + lots * (tl_i - prev_tl) * tl_mult
        spread_i = float(spread[i])
        beta_i = float(beta[i])
        z = math.nan
        if math.isfinite(spread_i) and etf_i > 0.0 and tl_i > 0.0 and beta_i > 0.0:
            mu_i = float(mu[i])
            sigma_i = float(sigma_eq[i])
            if sigma_i > 0.0 and math.isfinite(sigma_i) and math.isfinite(mu_i):
                z = (spread_i - mu_i) / sigma_i
                enter_i = float(z_enter[i])
                exit_i = float(z_exit[i])
                if lots == 0:
                    side = 0
                    if z > enter_i and allow_short:
                        side = 1
                    elif z < -enter_i and allow_long:
                        side = -1
                    if side != 0:
                        lots, shares, value = _enter(
                            z, etf_i, tl_i, beta_i, side, value,
                            max_capital, max_lots, etf_commission, tl_commission, tl_mult,
                        )
                elif abs(z) < exit_i:
                    value -= _fee(abs(shares), abs(lots), etf_i, tl_i, etf_commission, tl_commission, tl_mult)
                    lots = 0
                    shares = 0
                    n_trades += 1
        equity[i] = value
        if record:
            z_path[i] = z
            lots_path[i] = lots
        prev_etf = etf_i
        prev_tl = tl_i
        seen = True

    if lots != 0:
        value -= _fee(abs(shares), abs(lots), prev_etf, prev_tl, etf_commission, tl_commission, tl_mult)
        n_trades += 1
        equity[-1] = value
        lots = 0
        if record:
            lots_path[-1] = 0
    out = _performance(equity, times, n_trades, float(start_value))
    if record:
        out["z"] = z_path
        out["lots"] = lots_path
    return out


def _thresholds(value: float | np.ndarray, n: int, name: str) -> np.ndarray:
    """A positive finite threshold on every bar. A scalar is repeated."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        if not math.isfinite(float(arr)) or float(arr) <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")
        return np.full(n, float(arr), dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or length {n}, got shape {arr.shape}")
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr <= 0.0)):
        raise ValueError(f"{name} must be positive and finite on every bar")
    return arr


def _enter(
    z: float,
    etf: float,
    tl: float,
    beta: float,
    side: int,
    value: float,
    max_capital: float,
    max_lots: int,
    etf_commission: float,
    tl_commission: float,
    tl_mult: float,
) -> tuple[int, int, float]:
    lots = min(max_lots, max(1, int(round(abs(z)))))
    while lots >= 1:
        shares = max(1, int(round(lots * tl_mult / beta)))
        notional = shares * etf + lots * tl * tl_mult
        fee = _fee(shares, lots, etf, tl, etf_commission, tl_commission, tl_mult)
        if notional <= max_capital and value >= fee:
            signed_lots = lots if side > 0 else -lots
            signed_shares = -shares if side > 0 else shares
            return signed_lots, signed_shares, value - fee
        lots -= 1
    return 0, 0, value


def _fee(
    shares: int,
    lots: int,
    etf: float,
    tl: float,
    etf_commission: float,
    tl_commission: float,
    tl_mult: float,
) -> float:
    return shares * etf * etf_commission + lots * tl * tl_mult * tl_commission


def _performance(equity: np.ndarray, times: np.ndarray, n_trades: int, start_value: float) -> dict:
    """Absolute return, annualized return, Sharpe, and max drawdown on this equity curve."""
    series = pd.Series(equity, index=pd.to_datetime(times))
    end_value = float(series.iloc[-1])
    absolute_return = end_value / start_value - 1.0
    peak = series.cummax()
    max_drawdown = float(((series - peak) / peak).min())
    daily = series.resample("1D").last().dropna()
    n_days = int(daily.shape[0])
    first = float(daily.iloc[0] / start_value - 1.0)
    daily_return = pd.concat([pd.Series([first], index=daily.index[:1]), daily.pct_change().iloc[1:]])
    if n_days >= 1 and start_value > 0.0 and end_value > 0.0:
        annualized_return = float((end_value / start_value) ** (252 / n_days) - 1.0)
    else:
        annualized_return = float("nan")
    deviation = float(daily_return.std(ddof=1)) if len(daily_return) > 1 else 0.0
    if deviation > 0.0:
        sharpe_ratio = float(daily_return.mean() / deviation * math.sqrt(252))
    else:
        sharpe_ratio = float("nan")
    return {
        "absolute_return": absolute_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "trades": n_trades,
        "pnl": end_value - start_value,
        "start_value": start_value,
        "end_value": end_value,
        "n_days": n_days,
    }
