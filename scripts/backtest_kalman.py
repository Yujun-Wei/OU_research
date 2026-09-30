"""Out-of-sample backtest of a Kalman filter on the static hedge.

The filter estimates the AR(1) coefficients of the static spread. Its process
noise and the entry and exit bands are chosen by an in-sample Optuna search;
the chosen trial is then carried through the test. The test is not used to
choose the trial. R and P0 are the in-sample OLS residual variance and
coefficient covariance, and a is kept inside [0.001, 0.999]. The searched
filter parameters are log10(q_a) and log10(q_b).

Entry and exit are separate dynamic thresholds. On each minute,

    z_t = z0 * (half_life_t / half_life_bar) ** gamma_tau
              * (sigma_t / sigma_bar) ** gamma_sigma

half_life is ln(2) / theta and sigma is the diffusion coefficient, both from
that candidate's filter path. The bars are the medians of the in-sample
minute path, and the out-of-sample book keeps those same medians. z_entry is
floored at 1, z_exit at 0.05, and z_exit is pulled down so the gap is at
least 0.8. Zero elasticities leave a constant z0.

The training sessions are split chronologically into K=3 non-overlapping
subwindows of nearly equal length. Each subwindow book starts flat. The
objective is

    mean(S) - lambda * std(S)

with lambda = 1. std(S) is the standard deviation of those K Sharpes, with
divisor K. The full-window book is reported beside the objective and is not
what the search maximizes. The out-of-sample rows are reported for the chosen
candidate and are not used to choose it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.book import SIDES, Z_ENTER, Z_EXIT, print_metrics, simulate
from src.kalman import A_MAX, A_MIN, fit_ar1_ols, kalman_filter, minute_bars
from src.ols_fit import fit_ols
from src.static_hedge import fit_hedge, resample_last

MERGED_PATH = REPO_ROOT / "data" / "merged" / "511090_TL_20250701_20251230.parquet"
SPREAD = "x_static"
# Trial chosen by the training search at seed 0. The figures use this trial.
SELECTED = {
    "log10_qa": -5.099679150085213,
    "log10_qb": -5.650960798099461,
    "z0_entry": 2.421457866624639,
    "gamma_tau_entry": 0.5067726098098051,
    "gamma_sigma_entry": 0.4247093982734675,
    "z0_exit": 0.5084339002687808,
    "gamma_tau_exit": 0.0852586250997718,
    "gamma_sigma_exit": 0.18737078655070039,
}
K_SUBWINDOWS = 3
LAMBDA = 1.0
LOG10_QA = (-10.0, -5.0)
LOG10_QB = (-8.0, -3.0)
Z0_ENTRY = (1.2, 2.5)
Z0_EXIT = (0.1, 0.6)
GAMMA = (0.0, 0.6)
Z_ENTRY_MIN = 1.0
Z_EXIT_MIN = 0.05
Z_GAP = 0.8
N_TRIALS = 300
N_STARTUP = 40
LN2 = math.log(2.0)


def _objective(sharpes: np.ndarray, lam: float = LAMBDA) -> float:
    """mean(S) - lam * std(S). std uses divisor K, the length of S."""
    if sharpes.shape != (K_SUBWINDOWS,) or not np.all(np.isfinite(sharpes)):
        return float("nan")
    return float(sharpes.mean() - lam * sharpes.std(ddof=0))


def _objective_fields(sharpes: np.ndarray) -> dict[str, float]:
    finite = bool(np.all(np.isfinite(sharpes)))
    fields = {
        "objective": _objective(sharpes, LAMBDA),
        "sharpe_mean": float(sharpes.mean()) if finite else float("nan"),
        "sharpe_std": float(sharpes.std(ddof=0)) if finite else float("nan"),
    }
    for i, value in enumerate(sharpes, start=1):
        fields[f"sharpe_{i}"] = float(value)
    return fields


def _power_scale(level: np.ndarray, anchor: float, gamma: float) -> np.ndarray:
    """(level / anchor) ** gamma. A zero elasticity is identically one."""
    if gamma == 0.0:
        return np.ones(level.shape[0], dtype=np.float64)
    return np.power(level / anchor, gamma)


def _bands(
    half_life: np.ndarray,
    sigma: np.ndarray,
    tau_bar: float,
    sigma_bar: float,
    z0_entry: float,
    gamma_tau_entry: float,
    gamma_sigma_entry: float,
    z0_exit: float,
    gamma_tau_exit: float,
    gamma_sigma_exit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic entry and exit, then the floor and the minimum gap."""
    if tau_bar <= 0.0 or sigma_bar <= 0.0:
        raise ValueError(f"Anchors must be positive, got half-life {tau_bar}, sigma {sigma_bar}")
    if bool(np.any(half_life <= 0.0)) or bool(np.any(sigma <= 0.0)):
        raise ValueError("Half-life and sigma must be positive")
    z_entry = z0_entry * _power_scale(half_life, tau_bar, gamma_tau_entry) * _power_scale(sigma, sigma_bar, gamma_sigma_entry)
    z_exit = z0_exit * _power_scale(half_life, tau_bar, gamma_tau_exit) * _power_scale(sigma, sigma_bar, gamma_sigma_exit)
    z_entry = np.maximum(z_entry, Z_ENTRY_MIN)
    z_exit = np.maximum(z_exit, Z_EXIT_MIN)
    z_exit = np.minimum(z_exit, z_entry - Z_GAP)
    return z_entry, z_exit


def _check_bands() -> None:
    half = np.array([100.0, 200.0, 400.0])
    sigma = np.array([0.01, 0.02, 0.04])
    entry, exit_ = _bands(half, sigma, 200.0, 0.02, 2.0, 0.0, 0.0, 0.5, 0.0, 0.0)
    if not np.allclose(entry, 2.0) or not np.allclose(exit_, 0.5):
        raise RuntimeError("Zero elasticities must leave the baseline thresholds unchanged")
    entry, exit_ = _bands(half, sigma, 200.0, 0.02, 1.2, 0.0, 0.0, 0.6, 0.0, 0.0)
    if not np.allclose(entry, 1.2) or not np.allclose(exit_, 0.4):
        raise RuntimeError(f"Gap constraint failed, got entry {entry} exit {exit_}")
    entry, exit_ = _bands(np.array([1.0]), np.array([0.02]), 200.0, 0.02, 2.0, 1.0, 0.0, 0.5, 0.0, 0.0)
    if not np.allclose(entry, 1.0) or not np.allclose(exit_, 0.2):
        raise RuntimeError(f"Entry floor failed, got entry {entry} exit {exit_}")
    entry, exit_ = _bands(np.array([400.0]), np.array([0.02]), 200.0, 0.02, 2.0, 0.5, 0.0, 0.5, 0.0, 0.0)
    if not np.allclose(entry, 2.0 * math.sqrt(2.0)) or not np.allclose(exit_, 0.5):
        raise RuntimeError(f"Half-life elasticity failed, got entry {entry} exit {exit_}")


def _subwindow_masks(times: np.ndarray, k: int = K_SUBWINDOWS) -> list[np.ndarray]:
    """Chronological session blocks. A session is never split, and the blocks partition the ticks."""
    if k < 2:
        raise ValueError(f"Need at least 2 subwindows, got {k}")
    dates = np.asarray(times, dtype="datetime64[ns]").astype("datetime64[D]")
    sessions = np.unique(dates)
    n = int(sessions.size)
    if n < k:
        raise ValueError(f"Need at least {k} sessions to split, got {n}")
    sizes = np.full(k, n // k, dtype=int)
    sizes[: n % k] += 1
    edges = np.concatenate(([0], np.cumsum(sizes)))
    masks: list[np.ndarray] = []
    for start, end in zip(edges[:-1], edges[1:]):
        block = sessions[int(start):int(end)]
        mask = (dates >= block[0]) & (dates <= block[-1])
        if not bool(mask.any()):
            raise RuntimeError(f"Subwindow {len(masks) + 1} has no ticks")
        masks.append(mask)
    covered = np.logical_or.reduce(masks)
    if int(covered.sum()) != dates.size or any(
        bool(np.logical_and(masks[i], masks[j]).any()) for i in range(k) for j in range(i + 1, k)
    ):
        raise RuntimeError("Subwindows do not partition the ticks")
    return masks


def _tick_arrays(panel: pl.DataFrame, beta: float) -> dict[str, np.ndarray]:
    frame = (
        panel.select(
            "trade_time",
            "etf_last",
            "tl_last",
            pl.col(SPREAD).alias("spread"),
            pl.lit(beta).alias("beta"),
        )
        .drop_nulls(["spread", "etf_last", "tl_last"])
        .sort("trade_time")
    )
    return {
        "etf": frame["etf_last"].to_numpy().astype(np.float64),
        "tl": frame["tl_last"].to_numpy().astype(np.float64),
        "spread": frame["spread"].to_numpy().astype(np.float64),
        "beta": frame["beta"].to_numpy().astype(np.float64),
        "times": frame["trade_time"].to_numpy(),
    }


def _asof_index(minute_times: np.ndarray, tick_times: np.ndarray) -> np.ndarray:
    minute_ns = minute_times.astype("datetime64[ns]").astype(np.int64)
    tick_ns = tick_times.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(minute_ns, tick_ns, side="right") - 1
    if int(idx.min()) < 0:
        raise RuntimeError(f"{int((idx < 0).sum())} ticks fall before the first minute bar")
    return idx


def _masked(value: float | np.ndarray, mask: np.ndarray) -> float | np.ndarray:
    if np.asarray(value).ndim == 0:
        return value
    return np.asarray(value)[mask]


def _book_metrics(
    ticks: dict[str, np.ndarray],
    mu: np.ndarray,
    sigma_eq: np.ndarray,
    sides: str,
    z_enter: float | np.ndarray = Z_ENTER,
    z_exit: float | np.ndarray = Z_EXIT,
) -> dict:
    return simulate(
        ticks["etf"], ticks["tl"], ticks["spread"], ticks["beta"], mu, sigma_eq, ticks["times"],
        sides=sides, z_enter=z_enter, z_exit=z_exit,
    )


def _subwindow_sharpes(
    ticks: dict[str, np.ndarray],
    mu: np.ndarray,
    sigma_eq: np.ndarray,
    masks: list[np.ndarray],
    z_enter: float | np.ndarray = Z_ENTER,
    z_exit: float | np.ndarray = Z_EXIT,
) -> np.ndarray:
    """Sharpe of a flat-start symmetrical book on each subwindow, in chronological order."""
    sharpes = np.empty(len(masks), dtype=np.float64)
    for i, mask in enumerate(masks):
        metrics = simulate(
            ticks["etf"][mask], ticks["tl"][mask], ticks["spread"][mask], ticks["beta"][mask],
            mu[mask], sigma_eq[mask], ticks["times"][mask],
            sides="symmetrical", z_enter=_masked(z_enter, mask), z_exit=_masked(z_exit, mask),
        )
        sharpes[i] = metrics["sharpe_ratio"]
    return sharpes


def _score_windows(
    path_mu: np.ndarray,
    path_sigma: np.ndarray,
    idx: np.ndarray,
    ticks: dict[str, np.ndarray],
    masks: list[np.ndarray],
    z_enter: float | np.ndarray = Z_ENTER,
    z_exit: float | np.ndarray = Z_EXIT,
) -> dict:
    mu = path_mu[idx]
    sigma_eq = path_sigma[idx]
    sharpes = _subwindow_sharpes(ticks, mu, sigma_eq, masks, z_enter, z_exit)
    metrics = _book_metrics(ticks, mu, sigma_eq, "symmetrical", z_enter, z_exit)
    metrics.update(_objective_fields(sharpes))
    return metrics


def _score_trial(
    path,
    idx: np.ndarray,
    ticks: dict[str, np.ndarray],
    masks: list[np.ndarray],
    params: dict[str, float],
    anchors: tuple[float, float] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score one (Q, threshold) candidate. Anchors default to this path's in-sample medians."""
    theta = path.frame["theta_per_min"].to_numpy()
    sigma = path.frame["sigma"].to_numpy()
    half_life = LN2 / theta
    if anchors is None:
        tau_bar = float(np.median(half_life))
        sigma_bar = float(np.median(sigma))
    else:
        tau_bar, sigma_bar = anchors
    z_entry, z_exit = _bands(
        half_life[idx], sigma[idx], tau_bar, sigma_bar,
        params["z0_entry"], params["gamma_tau_entry"], params["gamma_sigma_entry"],
        params["z0_exit"], params["gamma_tau_exit"], params["gamma_sigma_exit"],
    )
    metrics = _score_windows(
        path.frame["mu"].to_numpy(), path.frame["sigma_eq"].to_numpy(), idx, ticks, masks, z_entry, z_exit,
    )
    metrics["qa"] = 10.0 ** params["log10_qa"]
    metrics["qb"] = 10.0 ** params["log10_qb"]
    metrics["tau_bar"] = tau_bar
    metrics["sigma_bar"] = sigma_bar
    metrics["z_entry_min"] = float(np.min(z_entry))
    metrics["z_entry_max"] = float(np.max(z_entry))
    metrics["z_exit_min"] = float(np.min(z_exit))
    metrics["z_exit_max"] = float(np.max(z_exit))
    metrics["n_clipped"] = path.n_clipped
    metrics["mu_min"] = float(path.frame["mu"].min())
    metrics["mu_max"] = float(path.frame["mu"].max())
    metrics["half_life_min"] = float(np.min(half_life))
    metrics["half_life_max"] = float(np.max(half_life))
    metrics["sigma_eq_min"] = float(path.frame["sigma_eq"].min())
    metrics["sigma_eq_max"] = float(path.frame["sigma_eq"].max())
    return metrics, z_entry, z_exit


def _fmt_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _print_objective(metrics: dict) -> None:
    print(
        f"  objective {_fmt_num(metrics['objective'])} = "
        f"mean {_fmt_num(metrics['sharpe_mean'])} - {LAMBDA:g} * std {_fmt_num(metrics['sharpe_std'])}, "
        f"subwindow sharpes "
        + ", ".join(_fmt_num(metrics[f"sharpe_{i}"]) for i in range(1, K_SUBWINDOWS + 1))
    )


def _print_subwindows(times: np.ndarray, masks: list[np.ndarray]) -> None:
    dates = np.asarray(times, dtype="datetime64[ns]").astype("datetime64[D]")
    print(
        f"Objective mean(S) - {LAMBDA:g} * std(S) on K={K_SUBWINDOWS} chronological subwindows "
        f"(std divisor {K_SUBWINDOWS})"
    )
    for i, mask in enumerate(masks, start=1):
        block = dates[mask]
        print(
            f"  subwindow {i}: {block[0]} .. {block[-1]} "
            f"({np.unique(block).size} sessions, {int(mask.sum()):,} ticks)"
        )


PARAM_KEYS = (
    "log10_qa",
    "log10_qb",
    "z0_entry",
    "gamma_tau_entry",
    "gamma_sigma_entry",
    "z0_exit",
    "gamma_tau_exit",
    "gamma_sigma_exit",
)
BASELINE = {
    "log10_qa": -5.0,
    "log10_qb": -6.0,
    "z0_entry": Z_ENTER,
    "gamma_tau_entry": 0.0,
    "gamma_sigma_entry": 0.0,
    "z0_exit": Z_EXIT,
    "gamma_tau_exit": 0.0,
    "gamma_sigma_exit": 0.0,
}
_TRIAL_ATTRS = (
    "objective",
    "sharpe_mean",
    "sharpe_std",
    "sharpe_1",
    "sharpe_2",
    "sharpe_3",
    "sharpe_ratio",
    "trades",
    "absolute_return",
    "annualized_return",
    "max_drawdown",
    "pnl",
    "n_days",
    "n_clipped",
    "mu_min",
    "mu_max",
    "half_life_min",
    "half_life_max",
    "sigma_eq_min",
    "sigma_eq_max",
    "qa",
    "qb",
    "tau_bar",
    "sigma_bar",
    "z_entry_min",
    "z_entry_max",
    "z_exit_min",
    "z_exit_max",
)


def _suggest(trial: optuna.Trial) -> dict[str, float]:
    return {
        "log10_qa": trial.suggest_float("log10_qa", *LOG10_QA),
        "log10_qb": trial.suggest_float("log10_qb", *LOG10_QB),
        "z0_entry": trial.suggest_float("z0_entry", *Z0_ENTRY),
        "gamma_tau_entry": trial.suggest_float("gamma_tau_entry", *GAMMA),
        "gamma_sigma_entry": trial.suggest_float("gamma_sigma_entry", *GAMMA),
        "z0_exit": trial.suggest_float("z0_exit", *Z0_EXIT),
        "gamma_tau_exit": trial.suggest_float("gamma_tau_exit", *GAMMA),
        "gamma_sigma_exit": trial.suggest_float("gamma_sigma_exit", *GAMMA),
    }


def _trial_row(trial: optuna.trial.FrozenTrial) -> dict:
    row = {"number": trial.number}
    row.update(trial.params)
    row.update(trial.user_attrs)
    return row


def main() -> None:
    _check_bands()
    panel = pl.read_parquet(MERGED_PATH)
    train = panel.filter(pl.col("in_sample"))
    train_end = train["trade_time"].dt.date().max()
    bars = resample_last(panel, "1m")
    hedge = fit_hedge(bars, train_end, "1min")
    print(
        f"Train {train['trade_time'].min()} .. {train['trade_time'].max()} "
        f"({train['trade_time'].dt.date().n_unique()} sessions through {train_end})"
    )
    minute = minute_bars(train, SPREAD)
    ols = fit_ar1_ols(minute, SPREAD)
    train_ou = fit_ols(train, SPREAD)
    if abs(ols.a - train_ou.a) > 1e-12 or abs(ols.b - train_ou.b) > 1e-12:
        raise RuntimeError(f"AR(1) OLS does not match fit_ols (a {ols.a} vs {train_ou.a}, b {ols.b} vs {train_ou.b})")
    if abs(ols.sigma_eps - train_ou.sigma_eps) > 1e-12:
        raise RuntimeError(f"sigma_eps {ols.sigma_eps} does not match fit_ols {train_ou.sigma_eps}")
    print(
        f"OLS a {ols.a:.6f}, b {ols.b:.6e}, sigma_eps {ols.sigma_eps:.6e}, "
        f"R {ols.R:.6e}, pairs {ols.n_pairs}"
    )
    print(f"P0 [[{ols.p_aa:.6e}, {ols.p_ab:.6e}], [{ols.p_ab:.6e}, {ols.p_bb:.6e}]]")
    print(f"a clipped to [{A_MIN}, {A_MAX}]")

    ticks = _tick_arrays(train, hedge.beta)
    masks = _subwindow_masks(ticks["times"])
    _print_subwindows(ticks["times"], masks)
    train_mu = np.full(ticks["etf"].shape[0], train_ou.mu)
    train_sigma = np.full(ticks["etf"].shape[0], train_ou.sigma_eq)
    train_sharpes = _subwindow_sharpes(ticks, train_mu, train_sigma, masks)
    print("In-sample book with the training-window OU fit held fixed")
    _print_objective(_objective_fields(train_sharpes))
    for sides in SIDES:
        print_metrics(f"train {sides}", _book_metrics(ticks, train_mu, train_sigma, sides))

    probed = kalman_filter(minute, SPREAD, ols, qa=0.0, qb=0.0)
    if probed.n_updates != ols.n_pairs:
        raise RuntimeError(f"Filter updates {probed.n_updates} do not match OLS pairs {ols.n_pairs}")
    idx = _asof_index(probed.frame["trade_time"].to_numpy(), ticks["times"])
    print(f"Minute bars {minute.height}, updates {probed.n_updates}, in-sample ticks {ticks['etf'].shape[0]:,}")

    fixed = kalman_filter(minute, SPREAD, ols, 10.0 ** BASELINE["log10_qa"], 10.0 ** BASELINE["log10_qb"])
    fixed_metrics, _, _ = _score_trial(fixed, idx, ticks, masks, BASELINE)
    scalar = _score_windows(fixed.frame["mu"].to_numpy(), fixed.frame["sigma_eq"].to_numpy(), idx, ticks, masks)
    if abs(fixed_metrics["objective"] - scalar["objective"]) > 1e-6:
        raise RuntimeError(
            f"Zero elasticities scored {fixed_metrics['objective']} against the fixed 2/0.5 book {scalar['objective']}"
        )
    print("Zero elasticities reproduce the fixed entry 2 / exit 0.5 book at the enqueued Q")
    _print_objective(fixed_metrics)

    def _rank(row: dict) -> float:
        value = row["objective"]
        return value if math.isfinite(value) else -math.inf

    def _objective_trial(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        path = kalman_filter(minute, SPREAD, ols, 10.0 ** params["log10_qa"], 10.0 ** params["log10_qb"])
        metrics, _, _ = _score_trial(path, idx, ticks, masks, params)
        for key in _TRIAL_ATTRS:
            trial.set_user_attr(key, metrics[key])
        value = metrics["objective"]
        return value if math.isfinite(value) else -1.0e9

    def _on_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            print(f"  trial {trial.number + 1}/{N_TRIALS} {trial.state.name}", flush=True)
            return
        params = trial.params
        attrs = trial.user_attrs
        print(
            f"  trial {trial.number + 1}/{N_TRIALS} objective {_fmt_num(attrs['objective'])}, "
            f"S [{_fmt_num(attrs['sharpe_1'])}, {_fmt_num(attrs['sharpe_2'])}, {_fmt_num(attrs['sharpe_3'])}], "
            f"z0 {_fmt_num(params['z0_entry'], 2)}/{_fmt_num(params['z0_exit'], 2)}, "
            f"gamma tau {_fmt_num(params['gamma_tau_entry'], 2)}/{_fmt_num(params['gamma_tau_exit'], 2)} "
            f"sigma {_fmt_num(params['gamma_sigma_entry'], 2)}/{_fmt_num(params['gamma_sigma_exit'], 2)}, "
            f"log10 qa {params['log10_qa']:.2f} qb {params['log10_qb']:.2f}",
            flush=True,
        )

    print(
        f"Optuna TPE, {N_TRIALS} trials, {N_STARTUP} random. "
        f"log10(qa) {LOG10_QA[0]}..{LOG10_QA[1]}, log10(qb) {LOG10_QB[0]}..{LOG10_QB[1]}, "
        f"z0 entry {Z0_ENTRY[0]}..{Z0_ENTRY[1]}, z0 exit {Z0_EXIT[0]}..{Z0_EXIT[1]}, "
        f"gamma {GAMMA[0]}..{GAMMA[1]}. "
        f"Floors entry {Z_ENTRY_MIN:g} exit {Z_EXIT_MIN:g}, gap {Z_GAP:g}."
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=N_STARTUP),
    )
    study.enqueue_trial(BASELINE)
    study.optimize(_objective_trial, n_trials=N_TRIALS, callbacks=[_on_trial])

    rows = [
        _trial_row(trial)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    print()
    print("Top 5")
    for row in sorted(rows, key=_rank, reverse=True)[:5]:
        print(
            f"  objective {_fmt_num(row['objective'])}, "
            f"z0 {_fmt_num(row['z0_entry'], 2)}/{_fmt_num(row['z0_exit'], 2)}, "
            f"gamma tau {_fmt_num(row['gamma_tau_entry'], 2)}/{_fmt_num(row['gamma_tau_exit'], 2)} "
            f"sigma {_fmt_num(row['gamma_sigma_entry'], 2)}/{_fmt_num(row['gamma_sigma_exit'], 2)}, "
            f"log10 qa {row['log10_qa']:.2f} qb {row['log10_qb']:.2f}"
        )

    winner = max(rows, key=_rank)
    params = {key: float(winner[key]) for key in PARAM_KEYS}
    print()
    print(
        f"Best in sample: log10(qa) {params['log10_qa']:.2f}, log10(qb) {params['log10_qb']:.2f} "
        f"(qa {winner['qa']:.3e}, qb {winner['qb']:.3e}), "
        f"z0 entry {params['z0_entry']:.3f} exit {params['z0_exit']:.3f}, "
        f"gamma tau {params['gamma_tau_entry']:.3f}/{params['gamma_tau_exit']:.3f}, "
        f"gamma sigma {params['gamma_sigma_entry']:.3f}/{params['gamma_sigma_exit']:.3f}"
    )
    _print_objective(winner)
    print_metrics("symmetrical full window", winner)
    print(
        f"  realized entry {winner['z_entry_min']:.3f} .. {winner['z_entry_max']:.3f}, "
        f"exit {winner['z_exit_min']:.3f} .. {winner['z_exit_max']:.3f}, "
        f"anchors half-life {winner['tau_bar']:.1f} min, sigma {winner['sigma_bar']:.4f}"
    )
    print(
        f"  path mu {winner['mu_min']:.4f} .. {winner['mu_max']:.4f}, "
        f"half-life {winner['half_life_min']:.1f} .. {winner['half_life_max']:.1f} min, "
        f"sigma_eq {winner['sigma_eq_min']:.4f} .. {winner['sigma_eq_max']:.4f}, "
        f"clipped updates {winner['n_clipped']}"
    )

    chosen = kalman_filter(minute, SPREAD, ols, float(winner["qa"]), float(winner["qb"]))
    _, z_entry, z_exit = _score_trial(chosen, idx, ticks, masks, params)
    mu = chosen.frame["mu"].to_numpy()
    sigma_eq = chosen.frame["sigma_eq"].to_numpy()
    print("Best candidate, in sample, by side")
    for sides in SIDES:
        print_metrics(sides, _book_metrics(ticks, mu[idx], sigma_eq[idx], sides, z_entry, z_exit))

    full_minute = minute_bars(panel, SPREAD)
    oos_path = kalman_filter(full_minute, SPREAD, ols, float(winner["qa"]), float(winner["qb"]))
    oos_ticks = _tick_arrays(panel.filter(~pl.col("in_sample")), hedge.beta)
    oos_idx = _asof_index(oos_path.frame["trade_time"].to_numpy(), oos_ticks["times"])
    oos_half = LN2 / oos_path.frame["theta_per_min"].to_numpy()
    oos_diffusion = oos_path.frame["sigma"].to_numpy()
    z_entry_oos, z_exit_oos = _bands(
        oos_half[oos_idx], oos_diffusion[oos_idx],
        float(winner["tau_bar"]), float(winner["sigma_bar"]),
        params["z0_entry"], params["gamma_tau_entry"], params["gamma_sigma_entry"],
        params["z0_exit"], params["gamma_tau_exit"], params["gamma_sigma_exit"],
    )
    oos_mu = oos_path.frame["mu"].to_numpy()
    oos_sigma = oos_path.frame["sigma_eq"].to_numpy()
    print(
        f"Out of sample, same Q and thresholds, in-sample anchors, filter carried forward "
        f"({panel.filter(~pl.col('in_sample'))['trade_time'].dt.date().n_unique()} sessions). "
        f"Not used to choose the candidate."
    )
    print(
        f"  realized entry {float(np.min(z_entry_oos)):.3f} .. {float(np.max(z_entry_oos)):.3f}, "
        f"exit {float(np.min(z_exit_oos)):.3f} .. {float(np.max(z_exit_oos)):.3f}"
    )
    for sides in SIDES:
        print_metrics(sides, _book_metrics(oos_ticks, oos_mu[oos_idx], oos_sigma[oos_idx], sides, z_entry_oos, z_exit_oos))


if __name__ == "__main__":
    main()
