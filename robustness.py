"""Multi-seed robustness sweep for the Monte Carlo backtest.

Every result the phased fixes produced so far rests on a single random seed (42)
and only ~25 trades, so a good showing could be luck. This script re-runs the
SAME backtest — imported from gbm_core, not reimplemented — across many seeds and
reports the spread of the headline metrics, so we can see which results are stable
and which are seed-dependent noise.

It measures only the backtest (not the forward chart). To keep each seed
comparable to the committed single-seed run, it consumes one throwaway forward
draw per ticker before that ticker's backtest, mirroring exactly the RNG the chart
script spends on its live Monte Carlo. That makes seed 42 here reproduce the
committed backtest numbers, which doubles as a correctness check on this harness.
"""

import numpy as np
import pandas as pd

from gbm_core import load_prices, run_monte_carlo, backtest

# --- config: must match Brownian_Motion_1st_Iteration.py ---
tickers = ["AAPL", "NVDA", "AMZN", "MSFT", "GLD"]
n = 100          # steps
T = 1            # years
M = 1000         # sims
BACKTEST_YEARS = range(2018, 2026)
SEEDS = range(50)   # 0..49; seed 42 falls in here and reproduces the committed run


def prepare_data():
    """Load closes for each ticker and SPY, and build the per-year, training-window
    correlation multiplier. All of this is deterministic (seed-independent)."""
    all_closes = {}
    all_returns = {}
    for ticker in tickers:
        data = load_prices(ticker)
        closes = data["Close"]
        all_closes[ticker] = closes.squeeze()
        all_returns[ticker] = np.log(closes / closes.shift(1)).squeeze()

    returns_df = pd.DataFrame(all_returns)
    backtest_multiplier = {}
    for year in BACKTEST_YEARS:
        window_corr = returns_df[f"{year-2}-01-01":f"{year}-01-01"].corr()
        backtest_multiplier[year] = {
            t: 1 - window_corr[t].drop(t).mean() for t in tickers
        }

    spy_closes = load_prices("SPY")["Close"].squeeze()
    return all_closes, spy_closes, backtest_multiplier


def run_one_seed(seed, all_closes, spy_closes, backtest_multiplier):
    """Run the full backtest for every ticker under one seed and return the
    portfolio-level tallies plus the per-ticker accuracy."""
    np.random.seed(seed)
    port_correct = 0
    port_total = 0
    port_pnl = 0.0
    port_invested = 0.0
    n_trades = 0
    per_ticker_acc = {}
    for ticker in tickers:
        # mirror the chart script's live Monte Carlo draw so the RNG stream lines
        # up with the committed single-seed run (the values are discarded)
        np.random.normal(0, np.sqrt(T / n), size=(M, n))
        result = backtest(ticker, all_closes[ticker], spy_closes,
                          backtest_multiplier, BACKTEST_YEARS, T, n, M,
                          verbose=False)
        port_correct += result["correct"]
        port_total += result["total"]
        port_pnl += result["total_pnl"]
        port_invested += result["total_invested"]
        n_trades += len(result["traded_returns"])
        per_ticker_acc[ticker] = result["correct"] / result["total"]
    return {
        "accuracy": port_correct / port_total,
        "correct": port_correct,
        "total": port_total,
        "return": port_pnl / port_invested if port_invested else 0.0,
        "pnl": port_pnl,
        "invested": port_invested,
        "n_trades": n_trades,
        "per_ticker_acc": per_ticker_acc,
    }


def describe(values):
    """mean, std, min, p25, median, p75, max of a list of numbers."""
    a = np.array(values, dtype=float)
    return {
        "mean": a.mean(),
        "std": a.std(ddof=1) if len(a) > 1 else 0.0,
        "min": a.min(),
        "p25": np.percentile(a, 25),
        "median": np.percentile(a, 50),
        "p75": np.percentile(a, 75),
        "max": a.max(),
    }


def main():
    all_closes, spy_closes, backtest_multiplier = prepare_data()

    runs = [run_one_seed(s, all_closes, spy_closes, backtest_multiplier) for s in SEEDS]

    # correctness check: seed 42 should reproduce the committed backtest numbers
    seed42 = next((r for s, r in zip(SEEDS, runs) if s == 42), None)
    print(f"{'='*60}")
    print("MULTI-SEED ROBUSTNESS SWEEP")
    print(f"{'='*60}")
    print(f"Seeds: {len(list(SEEDS))} ({min(SEEDS)}..{max(SEEDS)})   "
          f"Tickers: {len(tickers)}   Sims/path: {M}   Backtest years: "
          f"{BACKTEST_YEARS[0]}-{BACKTEST_YEARS[-1]}")
    if seed42 is not None:
        print(f"Seed-42 check (should match committed run): "
              f"accuracy {seed42['correct']}/{seed42['total']} = "
              f"{seed42['accuracy']:.0%}, portfolio return {seed42['return']:.1%}")
    print()

    metrics = {
        "Accuracy (%)":        [r["accuracy"] * 100 for r in runs],
        "Portfolio return (%)": [r["return"] * 100 for r in runs],
        "Portfolio P&L ($)":   [r["pnl"] for r in runs],
        "Trades (count)":      [r["n_trades"] for r in runs],
    }
    header = f"{'metric':<22}{'mean':>9}{'std':>9}{'min':>9}{'p25':>9}{'median':>9}{'p75':>9}{'max':>9}"
    print(header)
    print("-" * len(header))
    for name, vals in metrics.items():
        d = describe(vals)
        print(f"{name:<22}{d['mean']:>9.2f}{d['std']:>9.2f}{d['min']:>9.2f}"
              f"{d['p25']:>9.2f}{d['median']:>9.2f}{d['p75']:>9.2f}{d['max']:>9.2f}")
    print()

    # how often the strategy clears the two bars that matter
    acc = np.array([r["accuracy"] for r in runs])
    ret = np.array([r["return"] for r in runs])
    print(f"Share of seeds with accuracy > 50%: {(acc > 0.5).mean():.0%}")
    print(f"Share of seeds with positive return: {(ret > 0).mean():.0%}")
    print()

    # per-ticker accuracy stability
    print(f"{'per-ticker accuracy':<12}{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
    print("-" * 48)
    for ticker in tickers:
        vals = [r["per_ticker_acc"][ticker] * 100 for r in runs]
        d = describe(vals)
        print(f"{ticker:<12}{d['mean']:>9.1f}{d['std']:>9.1f}{d['min']:>9.1f}{d['max']:>9.1f}")


if __name__ == "__main__":
    main()
