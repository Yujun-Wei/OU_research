"""Align 511090 and TL last prices onto one clock for the overlapping sessions.

ETF snapshots are about 3 seconds; TL snapshots are about 0.5 seconds. The merged
panel uses the ETF timestamp. Each row takes the latest TL last at or before that
timestamp, within a short tolerance, and only inside the continuous session both
markets share: 09:30-11:30 and 13:00-15:00.

`etf_last` is the traded last with the cash dividend added back from the ex-date
on, so a position marked on it does not book the distribution as a price change.
`etf_last_raw` keeps the traded print. The cash amount is 1.50 per share. The
ex-date is the session whose `pre_close` sits more than half a point away from
the previous session's last print.
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ETF_DIR = REPO_ROOT / "data" / "国债ETF数据" / "3秒快照" / "511090"
DEFAULT_TL_PATH = REPO_ROOT / "data" / "国债期货合约数据" / "TL_main_snapshot.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"

START = date(2025, 7, 1)
END = date(2025, 12, 30)
TOLERANCE = "3s"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S%.f"
DIVIDEND = 1.50
# Ordinary pre_close-versus-prior-last gaps stay under 0.05.
GAP_FLAG = 0.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("merge_pair")


def in_overlap_session(time_col: str = "trade_time") -> pl.Expr:
    """Continuous hours both venues are actually trading.

    Morning is closed on the right so the 11:30 print is not carried into lunch.
    Afternoon stops at 15:00 because the ETF is done; TL keeps trading until 15:15.
    """
    tod = pl.col(time_col).dt.time()
    morning = tod.is_between(pl.time(9, 30, 0), pl.time(11, 30, 0), closed="left")
    afternoon = tod.is_between(pl.time(13, 0, 0), pl.time(15, 0, 0), closed="left")
    return morning | afternoon


def _parse_time(column: str = "trade_time") -> pl.Expr:
    return pl.col(column).cast(pl.String).str.to_datetime(TIME_FORMAT)


def _in_window(column: str, start: date, end: date) -> pl.Expr:
    return pl.col(column).dt.date().is_between(start, end, closed="both")


def _ex_date(quotes: pl.DataFrame) -> date | None:
    """The session where `pre_close` jumps away from the previous last print."""
    sessions = (
        quotes.group_by(pl.col("trade_time").dt.date().alias("session"), maintain_order=True)
        .agg(pl.col("pre_close").first(), pl.col("last").last())
        .sort("session")
        .with_columns((pl.col("pre_close") - pl.col("last").shift(1)).alias("gap"))
    )
    flagged = sessions.filter(pl.col("gap").abs() > GAP_FLAG)
    if flagged.is_empty():
        return None
    if flagged.height != 1:
        LOGGER.error("Expected one ETF ex-date, found %s", flagged.height)
        sys.exit(1)
    found = flagged["session"][0]
    if isinstance(found, date):
        return found
    return date.fromisoformat(str(found))


def load_etf(etf_dir: Path, start: date, end: date) -> pl.DataFrame:
    paths = sorted(etf_dir.glob("*.csv"))
    if not paths:
        LOGGER.error("No ETF snapshots in %s", etf_dir)
        sys.exit(2)

    LOGGER.info("Scanning %s ETF files from %s", len(paths), etf_dir)
    quotes = (
        pl.scan_csv(
            [str(path) for path in paths],
            schema_overrides={"trade_time": pl.String, "pre_close": pl.Float64, "last": pl.Float64},
        )
        .select("trade_time", "pre_close", "last")
        .with_columns(_parse_time())
        .filter(_in_window("trade_time", start, end))
        .filter(in_overlap_session())
        .filter(pl.col("last") > 0)
        .unique(subset=["trade_time"], keep="last")
        .sort("trade_time")
        .collect()
    )
    ex_date = _ex_date(quotes)
    if ex_date is None:
        LOGGER.info("No ETF ex-date in this window; etf_last is the traded last")
        adjusted = pl.col("last")
    else:
        LOGGER.info("ETF dividend %.2f added to last from %s", DIVIDEND, ex_date)
        adjusted = pl.when(pl.col("trade_time").dt.date() >= ex_date).then(pl.col("last") + DIVIDEND).otherwise(pl.col("last"))
    return quotes.with_columns(
        adjusted.alias("etf_last"),
        pl.col("last").alias("etf_last_raw"),
    ).select("trade_time", "etf_last", "etf_last_raw")


def load_tl(tl_path: Path, start: date, end: date) -> pl.DataFrame:
    if not tl_path.is_file():
        LOGGER.error("TL snapshot not found: %s", tl_path)
        sys.exit(2)

    LOGGER.info("Scanning TL main snapshot %s", tl_path)
    return (
        pl.scan_csv(
            tl_path,
            schema_overrides={
                "code": pl.String,
                "trade_time": pl.String,
                "last": pl.Float64,
            },
        )
        .select("code", "trade_time", "last")
        .with_columns(_parse_time())
        .filter(_in_window("trade_time", start, end))
        .filter(in_overlap_session("trade_time"))
        .filter(pl.col("last") > 0)
        .select(
            pl.col("trade_time").alias("tl_time"),
            pl.col("last").alias("tl_last"),
            pl.col("code").alias("tl_code"),
        )
        .unique(subset=["tl_time"], keep="last")
        .sort("tl_time")
        .collect()
    )


def merge(etf: pl.DataFrame, tl: pl.DataFrame, tolerance: str) -> pl.DataFrame:
    aligned = etf.join_asof(
        tl,
        left_on="trade_time",
        right_on="tl_time",
        strategy="backward",
        tolerance=tolerance,
    )
    missed = aligned["tl_last"].null_count()
    if missed:
        LOGGER.info("Dropped %s ETF ticks with no TL quote within %s", missed, tolerance)
    columns = ["trade_time", "etf_last", "tl_time", "tl_last", "tl_code"]
    if "etf_last_raw" in aligned.columns:
        columns.insert(2, "etf_last_raw")
    return aligned.drop_nulls("tl_last").select(columns)


def run(etf_dir: Path, tl_path: Path, output: Path, start: date, end: date, tolerance: str) -> None:
    etf = load_etf(etf_dir, start, end)
    tl = load_tl(tl_path, start, end)
    LOGGER.info(
        "Session rows: ETF %s (%s days), TL %s (%s days)",
        etf.height,
        etf["trade_time"].dt.date().n_unique(),
        tl.height,
        tl["tl_time"].dt.date().n_unique(),
    )

    merged = merge(etf, tl, tolerance)
    if merged.is_empty():
        LOGGER.error("Merge produced no rows")
        sys.exit(1)

    lag = (pl.col("trade_time") - pl.col("tl_time")).dt.total_seconds()
    lag_stats = merged.select(
        lag.median().alias("median"),
        lag.max().alias("max"),
    ).row(0)
    LOGGER.info(
        "Aligned %s rows, %s days, %s to %s, TL lag median %ss max %ss",
        merged.height,
        merged["trade_time"].dt.date().n_unique(),
        merged["trade_time"].min(),
        merged["trade_time"].max(),
        lag_stats[0],
        lag_stats[1],
    )
    for code, n in merged["tl_code"].value_counts().sort("tl_code").iter_rows():
        LOGGER.info("Contract %s: %s rows", code, n)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output, compression="zstd")
    LOGGER.info("Wrote %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align 511090 and TL last prices.")
    parser.add_argument("--etf-dir", type=Path, default=DEFAULT_ETF_DIR)
    parser.add_argument("--tl-path", type=Path, default=DEFAULT_TL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=date.fromisoformat, default=START)
    parser.add_argument("--end", type=date.fromisoformat, default=END)
    parser.add_argument("--tolerance", default=TOLERANCE)
    args = parser.parse_args()

    run(args.etf_dir, args.tl_path, args.output, args.start, args.end, args.tolerance)
