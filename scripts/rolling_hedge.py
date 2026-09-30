"""Fit the 20-session rolling hedge and store it on the merged panel.

Coefficients on a session are fit on the previous 20 sessions of 1-minute
lasts. The opening 20 sessions reuse that first fit.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ols_fit import fit_ols
from src.rolling_hedge import HEDGE_LOOKBACK, attach_rolling_hedge, fit_rolling_hedge
from src.static_hedge import resample_last, write_panel

MERGED_PATH = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"
QUARTER_END = date(2025, 9, 30)


def main() -> None:
    panel = pl.read_parquet(MERGED_PATH)
    bars = resample_last(panel, "1m").with_columns(pl.col("trade_time").dt.date().alias("session"))
    coefs = fit_rolling_hedge(bars, HEDGE_LOOKBACK)
    quarter = coefs.filter(pl.col("session") <= QUARTER_END)
    print(
        f"Lookback {HEDGE_LOOKBACK} sessions, backfill through {coefs.filter(pl.col('backfill'))['session'].max()}, "
        f"quarter {quarter['session'].min()} .. {quarter['session'].max()} "
        f"({quarter.height} sessions, beta_rolling {quarter['beta_rolling'].min():.3f} .. {quarter['beta_rolling'].max():.3f})"
    )

    merged = attach_rolling_hedge(panel, coefs)
    write_panel(merged, MERGED_PATH)
    print(f"Wrote alpha_rolling, beta_rolling, and x_rolling to {MERGED_PATH} ({merged.height} rows)")
    print(pl.DataFrame([fit_ols(merged.filter(pl.col("in_sample")), "x_rolling")]))


if __name__ == "__main__":
    main()
