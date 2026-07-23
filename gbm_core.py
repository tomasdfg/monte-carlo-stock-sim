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
DATA_START = "2016-01-01"


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
    path = f"data/{ticker}_data.csv"
    if os.path.exists(path):
        file_age = datetime.today() - datetime.fromtimestamp(os.path.getmtime(path))
    else:
        file_age = None

    if file_age is not None and file_age.days < 1:
        return pd.read_csv(path, index_col=0, header=[0, 1])
    data = yf.download(ticker, start=DATA_START, auto_adjust=True)
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


def backtest(ticker, closes, spy_closes, backtest_multiplier, years, T, n, M,
             cursor=None, conn=None, today=None, verbose=True):
    """Walk each backtest year: train mu/sigma on the trailing window, simulate,
    size a trade when the signal fires, and record the realized outcome under the
    symmetric take-profit/stop-loss rule. Returns the tallies the caller needs for
    the Sharpe ratios and the portfolio summary.

    Set verbose=False for a quiet run (the multi-seed sweep), and leave cursor as
    None to skip the per-year database writes.
    """
    correct = 0
    total = 0
    total_pnl = 0
    total_invested = 0
    # realized return of each year we actually traded
    traded_returns = []
    # same, but every backtest year, with no-trade years carried as a flat 0.0
    all_year_returns = []
    # per-year dollar P&L and capital deployed (0.0 in no-trade years), so the
    # caller can build a capital-weighted portfolio equity curve for drawdown
    yearly_pnl = []
    yearly_invested = []

    for year in years:
        training_data = closes[f"{year-2}-01-01":f"{year}-01-01"]
        actual_data = closes[f"{year}-01-01":f"{year+1}-01-01"]
        # calc MU and SIGMA from the training window only
        log_returns_t = np.log(training_data / training_data.shift(1))
        mu_annual_t = log_returns_t.mean() * 252
        sigma_annual_t = log_returns_t.std() * np.sqrt(252)
        # drift: training-window historical mu ONLY. Today's analyst-implied return
        # and today's CAPM expected_return are unknowable at the backtest date, so
        # blending them in leaks future information (Phase 2 de-leak). The forward
        # projection still uses the full blend.
        mu_t = float(mu_annual_t)
        # volatility
        sigma_t = float(sigma_annual_t)
        # calculate Moving Averages and Signal MA
        ma_50_t = training_data.rolling(50).mean()
        ma_200_t = training_data.rolling(200).mean()
        ma_signal_t = "BUY" if ma_50_t.iloc[-1] > ma_200_t.iloc[-1] else "AVOID"
        # get starting price for year:
        Starting_Price_t = float(actual_data.iloc[0])
        # run Monte Carlo Sim for this year
        St_t = run_monte_carlo(Starting_Price_t, mu_t, sigma_t, T, n, M)
        final_prices_t = St_t[-1]
        # probability that price will be above the starting price
        probability_t = (final_prices_t > Starting_Price_t).mean()

        actual_return = actual_data.iloc[-1] - actual_data.iloc[0]
        # getting "SPY" MA so that I can adjust probability threshold for golden vs death cross
        spy_training_data = spy_closes[f"{year-2}-01-01":f"{year}-01-01"]
        spy_ma_50_t = spy_training_data.rolling(50).mean()
        spy_ma_200_t = spy_training_data.rolling(200).mean()
        threshold = 0.8 if spy_ma_200_t.iloc[-1] > spy_ma_50_t.iloc[-1] else 0.63
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
            # calculate profit/loss in dollars
            profit_loss = investment * realized_return
            traded_returns.append(realized_return)
            all_year_returns.append(realized_return)
            yearly_pnl.append(profit_loss)
            yearly_invested.append(investment)
            total_pnl += profit_loss
            total_invested += investment
            if verbose:
                print(f"{year}: Investment=${investment:.2f}, P&L=${profit_loss:.2f}, Return={realized_return:.1%}")
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
        "traded_returns": traded_returns,
        "all_year_returns": all_year_returns,
        "yearly_pnl": yearly_pnl,
        "yearly_invested": yearly_invested,
        "p_gain_hold": p_gain_hold,
    }
