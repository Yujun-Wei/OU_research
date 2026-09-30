"""Daily rolling OLS hedge with a causal lookback.

The coefficients on session t are fit on the previous `lookback` sessions of
1-minute lasts. The first `lookback` sessions reuse the fit of that opening
window. The stored columns are

    x_rolling = P_ETF - beta_rolling * P_TL - alpha_rolling
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.static_hedge import ols_alpha_beta

HEDGE_LOOKBACK = 20


def fit_rolling_hedge(bars: pl.DataFrame, lookback: int = HEDGE_LOOKBACK) -> pl.DataFrame:
    """`alpha_rolling` and `beta_rolling` applied on each session. `backfill` marks the opening window."""
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2, got {lookback}")
    dated = bars.with_columns(pl.col("trade_time").dt.date().alias("session")).sort("trade_time")
    sessions = dated.select("session").unique().sort("session")["session"].to_list()
    if len(sessions) <= lookback:
        raise ValueError(f"Need more than {lookback} sessions, got {len(sessions)}")

    by_session = {
        frame["session"][0]: (frame["etf_last"].to_numpy(), frame["tl_last"].to_numpy())
        for frame in dated.partition_by("session", maintain_order=True)
    }

    def fit(window: list[date]) -> tuple[float, float]:
        y = np.concatenate([by_session[session][0] for session in window])
        x = np.concatenate([by_session[session][1] for session in window])
        return ols_alpha_beta(y, x)

    opening = sessions[:lookback]
    alpha, beta = fit(opening)
    rows = [
        {"session": session, "alpha_rolling": alpha, "beta_rolling": beta, "backfill": True}
        for session in opening
    ]
    for i in range(lookback, len(sessions)):
        alpha, beta = fit(sessions[i - lookback : i])
        rows.append({"session": sessions[i], "alpha_rolling": alpha, "beta_rolling": beta, "backfill": False})
    return pl.DataFrame(rows)


def attach_rolling_hedge(panel: pl.DataFrame, coefs: pl.DataFrame) -> pl.DataFrame:
    """Add the session's `alpha_rolling`, `beta_rolling`, and `x_rolling`. Replaces them if present."""
    existing = [name for name in ("alpha", "beta", "alpha_rolling", "beta_rolling", "x", "x_rolling") if name in panel.columns]
    base = panel.drop(existing) if existing else panel
    keyed = base.with_columns(pl.col("trade_time").dt.date().alias("session"))
    attached = keyed.join(coefs.select("session", "alpha_rolling", "beta_rolling"), on="session", how="left")
    missed = attached["alpha_rolling"].null_count()
    if missed:
        raise ValueError(f"{missed} rows have no rolling coefficients")
    return (
        attached.with_columns(
            (
                pl.col("etf_last") - pl.col("beta_rolling") * pl.col("tl_last") - pl.col("alpha_rolling")
            ).alias("x_rolling")
        )
        .drop("session")
    )
