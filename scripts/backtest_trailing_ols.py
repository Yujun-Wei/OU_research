"""Out-of-sample backtest of trailing OLS mean-reversion estimates.

The test period is the second half of sessions. For the static hedge and the
rolling hedge, μ, σ, and θ are re-estimated by OLS every 180 completed
one-minute bars on a trailing five sessions. Each spread is traded short,
long, and symmetrical, with entry 2 and exit 0.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.book import MAX_CAPITAL, SIDES, arrays, print_metrics, simulate
from src.ols_fit import LOOKBACK_SESSIONS, REFRESH_MINUTES, attach_ols, trailing_schedule
from src.rolling_hedge import HEDGE_LOOKBACK, fit_rolling_hedge
from src.static_hedge import TRAIN_FRAC, fit_hedge, resample_last, session_cutoff

MERGED_PATH = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"
SPREADS = (("static", "x_static"), ("rolling", "x_rolling"))


def main() -> None:
    panel = pl.read_parquet(MERGED_PATH)
    bars = resample_last(panel, "1m")
    train_end = session_cutoff(bars, TRAIN_FRAC)
    hedge = fit_hedge(bars, train_end, "1min")
    coefs = fit_rolling_hedge(bars, HEDGE_LOOKBACK)
    marked = panel.with_columns(pl.col("trade_time").dt.date().alias("session")).join(
        coefs.select("session", "alpha_rolling", "beta_rolling"), on="session", how="left"
    )
    static_gap = float(
        (marked["etf_last"] - hedge.beta * marked["tl_last"] - hedge.alpha - marked["x_static"]).abs().max()
    )
    rolling_gap = float(
        (
            marked["etf_last"]
            - marked["beta_rolling"] * marked["tl_last"]
            - marked["alpha_rolling"]
            - marked["x_rolling"]
        )
        .abs()
        .max()
    )
    if static_gap > 1e-8 or rolling_gap > 1e-8:
        raise RuntimeError(f"stored spread does not match the hedge (static {static_gap}, rolling {rolling_gap})")

    betas: dict[str, float | str] = {"static": hedge.beta, "rolling": "beta_rolling"}
    rows: list[dict] = []
    oos = marked.filter(pl.col("trade_time").dt.date() > train_end)
    print(
        f"Out of sample {oos['trade_time'].min()} .. {oos['trade_time'].max()} "
        f"({oos['trade_time'].dt.date().n_unique()} sessions, train through {train_end})"
    )
    print(
        f"Trailing OLS every {REFRESH_MINUTES} trading minutes, "
        f"lookback {LOOKBACK_SESSIONS} sessions, z enter 2 / exit 0.5, "
        f"start value {MAX_CAPITAL:,.0f}"
    )
    for name, column in SPREADS:
        schedule = trailing_schedule(panel, column, train_end)
        carried = int(schedule["carried_forward"].sum())
        lookback_bars = int(schedule["lookback_bars"][0])
        print(
            f"{name}: {schedule.height} fits, lookback {lookback_bars} bars, "
            f"{carried} carried forward, "
            f"mu {schedule['mu'].min():.4f} .. {schedule['mu'].max():.4f}"
        )
        attached = attach_ols(oos, schedule)
        missing = int(attached["mu"].null_count())
        if missing:
            raise RuntimeError(f"{name}: {missing} out-of-sample bars have no parameters")
        data = arrays(attached, column, betas[name])
        for sides in SIDES:
            print(f"  running {name} {sides} ({data['etf'].shape[0]:,} bars)", flush=True)
            metrics = simulate(
                data["etf"], data["tl"], data["spread"], data["beta"],
                data["mu"], data["sigma_eq"], data["times"], sides=sides,
            )
            metrics["hedge"] = name
            metrics["sides"] = sides
            rows.append(metrics)
            print_metrics(f"{name} {sides}", metrics)

    print()
    print(f"{'Hedge':<10} {'Sides':<14} {'Absolute Return':>16} {'Annualized Return':>18} {'Sharpe Ratio':>13} {'Max Drawdown':>14} {'Trades':>8}")
    for row in rows:
        print(
            f"{row['hedge']:<10} {row['sides']:<14} "
            f"{row['absolute_return']:>16.4%} "
            f"{row['annualized_return']:>18.4%} "
            f"{row['sharpe_ratio']:>13.3f} "
            f"{row['max_drawdown']:>14.4%} "
            f"{row['trades']:>8}"
        )


if __name__ == "__main__":
    main()
