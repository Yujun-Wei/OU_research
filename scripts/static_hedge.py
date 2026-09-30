"""Fit the static hedge on 1-minute bars, check it at 5 minutes, and store it.

`x_static` is written onto the merged panel with the 1-minute alpha and beta.
The 5-minute fit is only a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ols_fit import fit_ols
from src.static_hedge import TRAIN_FRAC, attach_static_hedge, fit_hedge, resample_last, session_cutoff, write_panel

MERGED_PATH = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"
FREQUENCIES = (("1m", "1min"), ("5m", "5min"))


def main() -> None:
    panel = pl.read_parquet(MERGED_PATH)
    bars = {name: resample_last(panel, every) for every, name in FREQUENCIES}
    train_end = session_cutoff(bars["1min"], TRAIN_FRAC)
    print(f"In-sample sessions run through {train_end}")

    fits = [fit_hedge(bars[name], train_end, name) for _every, name in FREQUENCIES]
    print(pl.DataFrame(fits).select(
        "frequency",
        "train_start",
        "train_end",
        "n_train",
        "n_test",
        "alpha",
        "beta",
        "r_squared",
        "resid_std_train",
        "resid_mean_test",
        "resid_std_test",
    ))
    print(f"beta 1min {fits[0].beta:.6f}  5min {fits[1].beta:.6f}  difference {fits[0].beta - fits[1].beta:.6f}")

    fit = fits[0]
    merged = attach_static_hedge(panel, fit.alpha, fit.beta, train_end)
    write_panel(merged, MERGED_PATH)
    print(
        f"Wrote x_static and in_sample to {MERGED_PATH} "
        f"({merged.height} rows, alpha {fit.alpha:.6f}, beta {fit.beta:.6f})"
    )
    print(pl.DataFrame([fit_ols(merged.filter(pl.col("in_sample")), "x_static")]))


if __name__ == "__main__":
    main()
