"""Out-of-sample spread, z-score, and position figures for the stage 1 report.

Both books trade the static OLS spread. One re-estimates the OU parameters
by OLS on a trailing five sessions, every 180 minutes, and uses fixed bands
at 2 and 0.5. The other is the Kalman filter chosen on the training window.
Figures are one-minute lasts. Positive lots are a short spread.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.book import Z_ENTER, Z_EXIT, arrays, simulate
from src.kalman import fit_ar1_ols, kalman_filter, minute_bars
from src.ols_fit import attach_ols, fit_ols, trailing_schedule
from src.static_hedge import fit_hedge, resample_last

MERGED_PATH = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"
FIGURES = REPO_ROOT / "reports" / "figures"
QUARTER_END = date(2025, 9, 30)

TRAIL = "#1F4E79"
KALMAN = "#C4622D"
BAND = "#8A9099"
SPREAD = "#243038"


def _load_backtest():
    path = REPO_ROOT / "scripts" / "backtest_kalman.py"
    spec = importlib.util.spec_from_file_location("backtest_kalman", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minute_last(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [name for name in frame.columns if name != "trade_time"]
    return (
        frame.sort("trade_time")
        .group_by(pl.col("trade_time").dt.truncate("1m").alias("trade_time"), maintain_order=True)
        .agg([pl.col(name).last() for name in columns])
        .sort("trade_time")
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
        }
    )


def _dates(axis, times: np.ndarray) -> None:
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.set_xlim(times[0], times[-1])


def _view_limit(values: np.ndarray, cap: float) -> float:
    """A symmetric limit that keeps the body of the series and ignores a few spikes."""
    finite = values[np.isfinite(values)]
    body = float(np.quantile(np.abs(finite), 0.995))
    return float(min(cap, max(2.5, body * 1.15)))


def main() -> None:
    panel = pl.read_parquet(MERGED_PATH)
    train = panel.filter(pl.col("in_sample"))
    test = panel.filter(~pl.col("in_sample"))
    train_end = train["trade_time"].dt.date().max()
    hedge = fit_hedge(resample_last(panel, "1m"), train_end, "1min")
    anchor = fit_ols(train, "x_static")
    quarter = (
        panel.filter(pl.col("trade_time").dt.date() <= QUARTER_END)
        .select("beta_rolling")
        .unique()
    )
    print(
        f"Static hedge alpha {hedge.alpha:.6f} beta {hedge.beta:.6f}, "
        f"train through {train_end}, test {test['trade_time'].dt.date().min()} .. {test['trade_time'].dt.date().max()}"
    )
    print(
        f"Training-window OU a {anchor.a:.5f} b {anchor.b:.6e} theta {anchor.theta_per_min:.6f} "
        f"mu {anchor.mu:.4f} sigma {anchor.sigma:.4f} sigma_eq {anchor.sigma_eq:.4f} "
        f"half-life {anchor.half_life_min:.1f} min"
    )
    print(
        f"Rolling beta through {QUARTER_END}: {quarter['beta_rolling'].min():.3f} .. {quarter['beta_rolling'].max():.3f}"
    )

    backtest = _load_backtest()
    schedule = trailing_schedule(panel, "x_static", train_end)
    trail_data = arrays(attach_ols(test, schedule), "x_static", hedge.beta)
    trail = simulate(
        trail_data["etf"], trail_data["tl"], trail_data["spread"], trail_data["beta"],
        trail_data["mu"], trail_data["sigma_eq"], trail_data["times"],
        sides="symmetrical", record=True,
    )
    print(
        f"trailing symmetrical: return {trail['absolute_return']:.4%} "
        f"sharpe {trail['sharpe_ratio']:.3f} drawdown {trail['max_drawdown']:.4%} trades {trail['trades']}"
    )

    selected = backtest.SELECTED
    qa = 10.0 ** selected["log10_qa"]
    qb = 10.0 ** selected["log10_qb"]
    train_minute = minute_bars(train, "x_static")
    ols = fit_ar1_ols(train_minute, "x_static")
    train_path = kalman_filter(train_minute, "x_static", ols, qa, qb)
    train_half = backtest.LN2 / train_path.frame["theta_per_min"].to_numpy()
    tau_bar = float(np.median(train_half))
    sigma_bar = float(np.median(train_path.frame["sigma"].to_numpy()))
    minute = minute_bars(panel, "x_static")
    path = kalman_filter(minute, "x_static", ols, qa, qb)
    ticks = backtest._tick_arrays(test, hedge.beta)
    idx = backtest._asof_index(path.frame["trade_time"].to_numpy(), ticks["times"])
    half_life = backtest.LN2 / path.frame["theta_per_min"].to_numpy()
    diffusion = path.frame["sigma"].to_numpy()
    z_entry, z_exit = backtest._bands(
        half_life[idx], diffusion[idx], tau_bar, sigma_bar,
        selected["z0_entry"], selected["gamma_tau_entry"], selected["gamma_sigma_entry"],
        selected["z0_exit"], selected["gamma_tau_exit"], selected["gamma_sigma_exit"],
    )
    kalman = simulate(
        ticks["etf"],
        ticks["tl"],
        ticks["spread"],
        ticks["beta"],
        path.frame["mu"].to_numpy()[idx],
        path.frame["sigma_eq"].to_numpy()[idx],
        ticks["times"],
        sides="symmetrical",
        z_enter=z_entry,
        z_exit=z_exit,
        record=True,
    )
    print(
        f"kalman symmetrical: return {kalman['absolute_return']:.4%} "
        f"annualized {kalman['annualized_return']:.4%} sharpe {kalman['sharpe_ratio']:.3f} "
        f"drawdown {kalman['max_drawdown']:.4%} trades {kalman['trades']}"
    )
    print(
        f"chosen log10 qa {selected['log10_qa']:.2f} qb {selected['log10_qb']:.2f} "
        f"z0 {selected['z0_entry']:.3f}/{selected['z0_exit']:.3f} "
        f"gamma tau {selected['gamma_tau_entry']:.3f}/{selected['gamma_tau_exit']:.3f} "
        f"gamma sigma {selected['gamma_sigma_entry']:.3f}/{selected['gamma_sigma_exit']:.3f} "
        f"anchors half-life {tau_bar:.2f} sigma {sigma_bar:.4f}"
    )

    trail_ticks = pl.DataFrame(
        {
            "trade_time": trail_data["times"],
            "spread": trail_data["spread"],
            "z_trail": trail["z"],
            "lots_trail": trail["lots"],
        }
    ).with_columns(pl.col("trade_time").cast(pl.Datetime("us")))
    kalman_ticks = pl.DataFrame(
        {
            "trade_time": ticks["times"],
            "z_kalman": kalman["z"],
            "lots_kalman": kalman["lots"],
            "z_entry": z_entry,
            "z_exit": z_exit,
        }
    ).with_columns(pl.col("trade_time").cast(pl.Datetime("us")))
    minute_path = _minute_last(trail_ticks.join(kalman_ticks, on="trade_time", how="inner"))
    times = minute_path["trade_time"].to_numpy()
    z_trail = minute_path["z_trail"].to_numpy()
    z_kal = minute_path["z_kalman"].to_numpy()
    entry = minute_path["z_entry"].to_numpy()
    exit_ = minute_path["z_exit"].to_numpy()
    for sides in ("short", "long"):
        sided = simulate(
            ticks["etf"], ticks["tl"], ticks["spread"], ticks["beta"],
            path.frame["mu"].to_numpy()[idx], path.frame["sigma_eq"].to_numpy()[idx], ticks["times"],
            sides=sides, z_enter=z_entry, z_exit=z_exit,
        )
        print(
            f"kalman {sides}: return {sided['absolute_return']:.4%} "
            f"annualized {sided['annualized_return']:.4%} sharpe {sided['sharpe_ratio']:.3f} "
            f"drawdown {sided['max_drawdown']:.4%} trades {sided['trades']}"
        )
    print(
        f"minute bars {minute_path.height}, "
        f"z trail {np.nanmin(z_trail):.2f} at {times[int(np.nanargmin(z_trail))]} .. {np.nanmax(z_trail):.2f}, "
        f"z kalman {np.nanmin(z_kal):.2f} at {times[int(np.nanargmin(z_kal))]} .. {np.nanmax(z_kal):.2f}, "
        f"entry {entry.min():.2f} .. {entry.max():.2f}, "
        f"exit {exit_.min():.2f} .. {exit_.max():.2f}, "
        f"lots trail {minute_path['lots_trail'].min()} .. {minute_path['lots_trail'].max()}, "
        f"lots kalman {minute_path['lots_kalman'].min()} .. {minute_path['lots_kalman'].max()}"
    )
    print(
        f"share of minutes with |z_kalman| > 6: {np.mean(np.abs(z_kal) > 6):.4%}, "
        f"entry > 6: {np.mean(entry > 6):.4%}"
    )

    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_spread_z(times, minute_path, z_trail, z_kal, entry, exit_)
    _plot_position(times, minute_path)
    print(f"Wrote {FIGURES / 'spread_z.pdf'}")
    print(f"Wrote {FIGURES / 'position.pdf'}")


def _plot_spread_z(
    times: np.ndarray,
    minute_path: pl.DataFrame,
    z_trail: np.ndarray,
    z_kal: np.ndarray,
    entry: np.ndarray,
    exit_: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 7.4), sharex=True, constrained_layout=True)
    spread = minute_path["spread"].to_numpy()
    axes[0].plot(times, spread, color=SPREAD, lw=0.45)
    axes[0].axhline(0.0, color=BAND, lw=0.5)
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Static spread")

    axes[1].plot(times, z_trail, color=TRAIL, lw=0.45)
    for level in (Z_ENTER, Z_EXIT):
        axes[1].axhline(level, color=BAND, lw=0.6, ls="--")
        axes[1].axhline(-level, color=BAND, lw=0.6, ls="--")
    axes[1].axhline(0.0, color=BAND, lw=0.4)
    axes[1].set_ylabel("$z$")
    axes[1].set_title("Trailing OLS $z$-score")
    trail_lim = _view_limit(z_trail, cap=8.0)
    axes[1].set_ylim(-trail_lim, trail_lim)

    axes[2].plot(times, entry, color=BAND, lw=0.6, ls="--")
    axes[2].plot(times, -entry, color=BAND, lw=0.6, ls="--")
    axes[2].plot(times, exit_, color=BAND, lw=0.5, ls=":")
    axes[2].plot(times, -exit_, color=BAND, lw=0.5, ls=":")
    axes[2].plot(times, z_kal, color=KALMAN, lw=0.45)
    axes[2].axhline(0.0, color=BAND, lw=0.4)
    axes[2].set_ylabel("$z$")
    axes[2].set_title("Kalman $z$-score")
    kalman_lim = _view_limit(np.concatenate([z_kal, np.minimum(entry, 8.0)]), cap=8.0)
    axes[2].set_ylim(-kalman_lim, kalman_lim)
    _dates(axes[2], times)
    fig.savefig(FIGURES / "spread_z.pdf")
    fig.savefig("/tmp/stage1_spread_z.png", dpi=140)
    plt.close(fig)


def _plot_position(times: np.ndarray, minute_path: pl.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 5.2), sharex=True, constrained_layout=True)
    series = (
        (axes[0], minute_path["lots_trail"].to_numpy(), TRAIL, "Trailing OLS"),
        (axes[1], minute_path["lots_kalman"].to_numpy(), KALMAN, "Kalman filter"),
    )
    peak = max(abs(int(minute_path["lots_trail"].min())), abs(int(minute_path["lots_trail"].max())))
    peak = max(peak, abs(int(minute_path["lots_kalman"].min())), abs(int(minute_path["lots_kalman"].max())))
    for axis, lots, color, title in series:
        axis.fill_between(times, lots, 0.0, step="post", color=color, alpha=0.18, lw=0.0)
        axis.step(times, lots, where="post", color=color, lw=0.7)
        axis.axhline(0.0, color=BAND, lw=0.5)
        axis.set_ylabel("TL lots")
        axis.set_title(title)
        axis.set_ylim(-(peak + 1), peak + 1)
    _dates(axes[1], times)
    fig.savefig(FIGURES / "position.pdf")
    fig.savefig("/tmp/stage1_position.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
