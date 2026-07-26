"""Reusable core of the Monte Carlo backtest.

The chart-producing script (Brownian_Motion_1st_Iteration.py) and the multi-seed
robustness sweep (robustness.py) both import these functions, so the strategy has
exactly one implementation. Keeping the simulation and backtest logic here — rather
than duplicated between a live pass and a backtest pass — is what stopped the two
versions of the drift from silently diverging.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# pinned start of the price history. period="10y" anchored to today, so the
# dataset silently shifted every day and the 2018 training window was truncated
# (10y back from mid-2026 starts at mid-2016). With a fixed start, every backtest
# window is a date-bounded slice that never moves; the end stays open so the
# forward projection always uses the latest close.
DATA_START = "2010-01-01"

# where the CSV price cache and the SQLite database live. Gitignored, so it does
# not exist in a fresh clone and has to be created on demand.
DATA_DIR = "data"

# trading days in a one-year horizon. The bootstrap simulates a year by drawing
# this many daily returns; the GBM path count (n) is an abstract discretization
# and is not the right horizon for resampling real daily returns.
TRADING_DAYS = 252

# --- walk-forward configuration ---
# WINDOW_MODE picks the training scheme for each backtest year:
#   "rolling"   -> train on the trailing TRAINING_YEARS (a fixed-length window that
#                  rolls forward; adapts to the recent regime, forgets old data)
#   "expanding" -> anchor the window at DATA_START and grow it toward the test year
#                  (uses all history to date; more stable, slower to adapt)
TRAINING_YEARS = 2
WINDOW_MODE = "rolling"

# transaction cost / slippage charged per round-trip trade, as a fraction of the
# notional. 0.001 = a flat 0.1% per trade (a deliberately conservative one-sided
# figure; double it to model paying on both the entry and the exit).
TRANSACTION_COST = 0.001


def train_start_year(test_year):
    """First calendar year of the training window for a given test year, honoring
    WINDOW_MODE. Used for the return/vol estimation, the moving-average signal and
    the position-sizing correlation, so they all train on the same window."""
    if WINDOW_MODE == "expanding":
        return int(DATA_START[:4])
    return test_year - TRAINING_YEARS


def backtest_years(last_data_date):
    """Testable years given the available data: the first year that has a full
    training window behind it, through the last fully-elapsed calendar year."""
    first = int(DATA_START[:4]) + TRAINING_YEARS
    last = last_data_date.year - 1
    return range(first, last + 1)


# define a function that states how much to put in for each trade
def position_size(probability):
    if probability >= 0.80:
        return 5000
    elif probability >= 0.70:
        return 1500
    elif probability >= 0.63:
        return 500
    else:
        return 0


# a Sharpe needs at least two samples and some dispersion between them
def sharpe_ratio(returns, rf):
    if len(returns) < 2:
        return 0.0
    stdev = np.std(returns, ddof=1)
    if stdev == 0:
        return 0.0
    return (np.mean(returns) - rf) / stdev


def win_rate(traded_returns):
    """Fraction of trades that finished profitable. None when there were no
    trades, so the caller can render it as n/a rather than a misleading 0%."""
    if len(traded_returns) == 0:
        return None
    wins = sum(1 for r in traded_returns if r > 0)
    return wins / len(traded_returns)


def max_drawdown(period_returns):
    """Largest peak-to-trough decline of a $1 equity curve compounded from the
    given per-period returns. Returns <= 0 (0.0 if the curve never dips below a
    prior peak)."""
    if len(period_returns) == 0:
        return 0.0
    equity = np.cumprod(1 + np.asarray(period_returns, dtype=float))
    running_peak = np.maximum.accumulate(equity)
    return float((equity / running_peak - 1).min())


def load_prices(ticker):
    """Load a ticker's price history from its cached CSV when the cache is less
    than a day old, otherwise re-download it and refresh the cache. This caching
    block used to be copied out for the ticker loop and again for SPY."""
    path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if os.path.exists(path):
        file_age = datetime.today() - datetime.fromtimestamp(os.path.getmtime(path))
    else:
        file_age = None

    if file_age is not None and file_age.days < 1:
        return pd.read_csv(path, index_col=0, header=[0, 1])
    data = yf.download(ticker, start=DATA_START, auto_adjust=True)
    # data/ is gitignored, so it is absent on a fresh clone (e.g. a Streamlit Cloud
    # deploy). Create it here rather than relying on the chart script having run.
    os.makedirs(DATA_DIR, exist_ok=True)
    data.to_csv(path)
    return data


def run_monte_carlo(S0, mu, sigma, T, n, M):
    """Simulate M geometric-Brownian-motion price paths over n steps and return
    the (n+1, M) array of prices. This block used to be duplicated in the live
    forecast and the backtest, which is exactly how the backtest's drift quietly
    diverged from the live one."""
    dt = T / n
    steps = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M, n)).T
    )
    steps = np.vstack([np.ones(M), steps])
    return S0 * steps.cumprod(axis=0)


def simulate_correlated_portfolio(mu_vec, sigma_vec, corr, weights, T, n, M):
    """Simulate the joint 1-year evolution of a portfolio of correlated assets and
    return the M final portfolio values (portfolio starts at value 1.0).

    Correlation is imposed with a Cholesky factor of the correlation matrix: for
    independent standard normals Z, the product L @ Z has covariance L L^T = corr,
    so the simulated names move together the way they historically do. Simulating
    the assets independently would ignore this and understate portfolio risk,
    because these names tend to fall together in a sell-off.

    mu_vec / sigma_vec / weights are arrays aligned with the rows/cols of corr.
    """
    k = len(mu_vec)
    mu_vec = np.asarray(mu_vec, dtype=float)
    sigma_vec = np.asarray(sigma_vec, dtype=float)
    weights = np.asarray(weights, dtype=float)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # nudge onto the PSD cone if sampling noise made corr non-positive-definite
        L = np.linalg.cholesky(corr + 1e-10 * np.eye(k))
    dt = T / n
    # price relative of each asset, all starting at 1.0
    rel = np.ones((k, M))
    for _ in range(n):
        Z = np.random.normal(size=(k, M))
        correlated = L @ Z  # (k, M) standard normals carrying the target correlation
        rel *= np.exp(
            (mu_vec[:, None] - sigma_vec[:, None] ** 2 / 2) * dt
            + sigma_vec[:, None] * np.sqrt(dt) * correlated
        )
    # portfolio value = sum_i weight_i * (price relative of asset i)
    return weights @ rel


def bootstrap_paths(S0, daily_log_returns, n_days, M):
    """Simulate M price paths over n_days by resampling actual historical daily
    log returns with replacement (a bootstrap), instead of drawing Gaussian shocks.

    The empirical distribution carries the real drift, volatility, skew and fat
    tails of the input returns, so crash days and other extremes appear at their
    true historical frequency rather than being assumed away by a bell curve.
    Because we resample log returns directly, they sum in log space with no Ito
    -sigma^2/2 correction. Returns the (n_days+1, M) price array."""
    r = np.asarray(daily_log_returns, dtype=float)
    r = r[~np.isnan(r)]
    sampled = np.random.choice(r, size=(n_days, M), replace=True)
    log_paths = np.vstack([np.zeros(M), sampled.cumsum(axis=0)])
    return S0 * np.exp(log_paths)


def backtest(ticker, closes, spy_closes, backtest_multiplier, years, M,
             cursor=None, conn=None, today=None, verbose=True):
    """Walk each backtest year: bootstrap the trailing window's daily returns to
    simulate the year, size a trade when the signal fires, and record the realized
    outcome under the symmetric take-profit/stop-loss rule. Returns the tallies the
    caller needs for the Sharpe ratios and the portfolio summary.

    Set verbose=False for a quiet run (the multi-seed sweep), and leave cursor as
    None to skip the per-year database writes.
    """
    correct = 0
    total = 0
    total_pnl = 0
    total_invested = 0
    total_cost = 0.0
    # realized return of each year we actually traded
    traded_returns = []
    # same, but every backtest year, with no-trade years carried as a flat 0.0
    all_year_returns = []
    # per-year prediction detail, so a caller can render a year-by-year view
    # (e.g. the dashboard's accuracy grid) without re-deriving any of it
    yearly_detail = []
    # per-year dollar P&L and capital deployed (0.0 in no-trade years), so the
    # caller can build a capital-weighted portfolio equity curve for drawdown
    yearly_pnl = []
    yearly_invested = []

    for year in years:
        train_start = train_start_year(year)
        training_data = closes[f"{train_start}-01-01":f"{year}-01-01"]
        actual_data = closes[f"{year}-01-01":f"{year+1}-01-01"]
        # this window's daily log returns are the bootstrap source. Their mean is
        # the training-window historical drift (the Phase 2 de-leaked, no-look-ahead
        # drift), and resampling them keeps the window's real volatility, skew and
        # fat tails instead of assuming a normal. The forward projection still uses
        # the parametric blended-mu GBM.
        log_returns_t = np.log(training_data / training_data.shift(1))
        # calculate Moving Averages and Signal MA
        ma_50_t = training_data.rolling(50).mean()
        ma_200_t = training_data.rolling(200).mean()
        ma_signal_t = "BUY" if ma_50_t.iloc[-1] > ma_200_t.iloc[-1] else "AVOID"
        # get starting price for year:
        Starting_Price_t = float(actual_data.iloc[0])
        # simulate the year by bootstrapping TRADING_DAYS daily returns
        St_t = bootstrap_paths(Starting_Price_t, log_returns_t, TRADING_DAYS, M)
        final_prices_t = St_t[-1]
        # probability that price will be above the starting price
        probability_t = (final_prices_t > Starting_Price_t).mean()

        actual_return = actual_data.iloc[-1] - actual_data.iloc[0]
        # getting "SPY" MA so that I can adjust probability threshold for golden vs death cross
        spy_training_data = spy_closes[f"{train_start}-01-01":f"{year}-01-01"]
        spy_ma_50_t = spy_training_data.rolling(50).mean()
        spy_ma_200_t = spy_training_data.rolling(200).mean()
        threshold = 0.8 if spy_ma_200_t.iloc[-1] > spy_ma_50_t.iloc[-1] else 0.63
        # recorded in yearly_detail below; overwritten when a trade actually fires
        traded_this_year = False
        exit_reason_this_year = None
        # probabilty and buy signal
        if (probability_t > threshold) and (ma_signal_t == 'BUY'):
            if verbose:
                print(f"{year}: Strong Buy")
            #how much is put in for each trade
            investment = backtest_multiplier[year][ticker] * position_size(probability_t)
            # strength confidence signal
            if verbose:
                print(f"{year}: P={probability_t:.0%}, MA={ma_signal_t}, Investment=${investment}")
            # exit rule: +20% take-profit vs -20% stop-loss (symmetric barriers),
            # whichever the price touches FIRST chronologically (daily closes). The
            # symmetric -20% gives a trade room to ride out a correction and still
            # reach the take-profit rather than being stopped out early. Checking
            # only whether a level was ever touched would let a late take-profit
            # mask a stop-loss that already triggered earlier in the year.
            tp_level = Starting_Price_t * 1.20
            sl_level = Starting_Price_t * 0.80
            tp_days = actual_data.index[actual_data.to_numpy() >= tp_level]
            sl_days = actual_data.index[actual_data.to_numpy() <= sl_level]
            first_tp = tp_days[0] if len(tp_days) else None
            first_sl = sl_days[0] if len(sl_days) else None
            if first_tp is not None and (first_sl is None or first_tp <= first_sl):
                exit_reason = "take-profit"
                realized_return = 0.20
            elif first_sl is not None:
                exit_reason = "stop-loss"
                realized_return = -0.20
            else:
                exit_reason = "year-end"
                realized_return = actual_return / Starting_Price_t
            # charge the round-trip transaction cost against the trade: a marginal
            # winner can net negative once trading costs are paid.
            trade_cost = investment * TRANSACTION_COST
            net_return = realized_return - TRANSACTION_COST
            profit_loss = investment * net_return
            traded_returns.append(net_return)
            all_year_returns.append(net_return)
            yearly_pnl.append(profit_loss)
            yearly_invested.append(investment)
            total_pnl += profit_loss
            total_invested += investment
            total_cost += trade_cost
            traded_this_year = True
            exit_reason_this_year = exit_reason
            if verbose:
                print(f"{year}: Investment=${investment:.2f}, P&L=${profit_loss:.2f}, "
                      f"Net return={net_return:.1%} (gross {realized_return:.1%}, cost ${trade_cost:.2f})")
                print(f"Exit: {exit_reason}")
        else:
            if verbose:
                print(f"{year}: No Trade")
            all_year_returns.append(0.0)
            yearly_pnl.append(0.0)
            yearly_invested.append(0.0)

        #check what actually happened:
        predicted_direction = "UP" if probability_t > 0.5 else "DOWN"
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        is_correct = 1 if predicted_direction == actual_direction else 0
        yearly_detail.append({
            "ticker": ticker,
            "year": year,
            "predicted": predicted_direction,
            "actual": actual_direction,
            "probability": float(probability_t),
            "correct": is_correct,
            "traded": bool(traded_this_year),
            "exit": exit_reason_this_year,
        })
        if verbose:
            print(f"{year}: P(gain)={probability_t:.0%}, Predicted={predicted_direction}, Actual={actual_direction}")
        # credit any correct call, not just correct UP calls
        correct += is_correct
        total += 1

        # clear this run's prior row for the same ticker/year so reruns refresh it
        if cursor is not None:
            cursor.execute(
                "DELETE FROM backtest_results WHERE ticker = ? AND year = ? AND today = ?",
                (ticker, year, today),
            )
            cursor.execute('''
                INSERT INTO backtest_results
                (ticker, year, predicted_direction, actual_direction, probability_t, is_correct, today)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, year, predicted_direction, actual_direction, probability_t, is_correct, today))
            conn.commit()

    #calculating what the hold amount would be, over the same span as the backtest
    bh_first_year = years[0]
    bh_last_year = years[-1]
    bh_start = closes[f"{bh_first_year}-01-01" : f"{bh_first_year}-02-01"].iloc[0]
    bh_end = closes[f"{bh_last_year}-01-01" : f"{bh_last_year+1}-01-01"].iloc[-1]
    p_gain_hold = (bh_end / bh_start) - 1

    return {
        "correct": correct,
        "total": total,
        "total_pnl": total_pnl,
        "total_invested": total_invested,
        "total_cost": total_cost,
        "traded_returns": traded_returns,
        "all_year_returns": all_year_returns,
        "yearly_pnl": yearly_pnl,
        "yearly_invested": yearly_invested,
        "yearly_detail": yearly_detail,
        "p_gain_hold": p_gain_hold,
    }
