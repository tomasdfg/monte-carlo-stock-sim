import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
import plotly_viz as PLV

# --- SETTINGS ---
tickers = ["AAPL", "NVDA", "AMZN", "MSFT", "GLD"]
# number of steps
n = 100
# time in years
T = 1
# number of sims
M = 1000
# probability target as a % gain above each ticker's current price, so it is
# meaningful across very different price levels. A flat $300 target was ~30% for
# AAPL but nonsensical for GLD (~$60) or NVDA. Applied per ticker in PASS 2.
PRICE_TARGET_PCT = 0.30
# pinned start of the price history. period="10y" anchored to today, so the
# dataset silently shifted every day and the 2018 training window was truncated
# (10y back from mid-2026 starts at mid-2016). With a fixed start, every backtest
# window is a date-bounded slice that never moves; the end stays open so the
# forward projection always uses the latest close.
DATA_START = "2016-01-01"
# fixed seed so runs are reproducible and changes to the model can be compared
# against each other rather than against Monte Carlo sampling noise.
# set to None to draw a fresh sample each run.
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# --- DATA LOADING ---
today = datetime.today().strftime('%Y-%m-%d')
# create data folder
os.makedirs("data", exist_ok=True)

# create database and tables
conn = sqlite3.connect("data/market_data.db")
cursor = conn.cursor()

# Older versions of this script wrote `prices` with to_sql(if_exists="replace"), which
# blew away the schema below and left stringified MultiIndex columns. Drop it so the
# table always matches the schema we insert against; it is a cache rebuilt from the
# per-ticker CSVs on every run, so nothing unrecoverable is lost.
cursor.execute("DROP TABLE IF EXISTS prices")
cursor.execute('''
    CREATE TABLE prices (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (ticker, date)
    )
''')
conn.commit()
print("Database created successfully")

cursor.execute('''
    CREATE TABLE IF NOT EXISTS fundamentals (
               ticker TEXT,
               beta REAL,
               risk_free_rate REAL,
               analyst_target REAL,
               expected_return REAL,
               last_updated TEXT
    )
''')
conn.commit()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS backtest_results (
        ticker TEXT,
        year INTEGER,
        predicted_direction TEXT,
        actual_direction TEXT,
        probability_t REAL,
        is_correct INTEGER,
        today TEXT
    )
''')
conn.commit()

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
             cursor, conn, today):
    """Walk each backtest year: train mu/sigma on the trailing window, simulate,
    size a trade when the signal fires, and record the realized outcome under the
    symmetric take-profit/stop-loss rule. Returns the tallies the caller needs for
    the Sharpe ratios and the portfolio summary."""
    correct = 0
    total = 0
    total_pnl = 0
    total_invested = 0
    # realized return of each year we actually traded
    traded_returns = []
    # same, but every backtest year, with no-trade years carried as a flat 0.0
    all_year_returns = []

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
            print(f"{year}: Strong Buy")
            #how much is put in for each trade
            investment = backtest_multiplier[year][ticker] * position_size(probability_t)
            # strength confidence signal
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
            total_pnl += profit_loss
            total_invested += investment
            print(f"{year}: Investment=${investment:.2f}, P&L=${profit_loss:.2f}, Return={realized_return:.1%}")
            print(f"Exit: {exit_reason}")
        else:
            print(f"{year}: No Trade")
            all_year_returns.append(0.0)

        #check what actually happened:
        predicted_direction = "UP" if probability_t > 0.5 else "DOWN"
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        is_correct = 1 if predicted_direction == actual_direction else 0
        print(f"{year}: P(gain)={probability_t:.0%}, Predicted={predicted_direction}, Actual={actual_direction}")
        # credit any correct call, not just correct UP calls
        correct += is_correct
        total += 1

        # clear this run's prior row for the same ticker/year so reruns refresh it
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
        "p_gain_hold": p_gain_hold,
    }

plt.style.use("dark_background") # sets background to black
fig, axes = plt.subplots(2, 3, figsize=(20,7.25)) # Sets sizing but conflicts with plt.tight_layout()
axes = axes.flatten() # converts 2D grid to a simple list

all_returns = {} # this makes a dictionary to store all the returns from each ticker in one place
all_closes = {}

##### PASS 1 #####
for ticker in tickers:
    data = load_prices(ticker)

    # save downloaded price history for {ticker} to the prices table in the database.
    # yfinance hands back MultiIndex columns like ("Close", "AAPL"); flatten them into
    # the flat schema above and tag each row with its ticker, so every ticker coexists.
    price_rows = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime(data.index).strftime("%Y-%m-%d"),
        "open": data["Open"].squeeze().to_numpy(),
        "high": data["High"].squeeze().to_numpy(),
        "low": data["Low"].squeeze().to_numpy(),
        "close": data["Close"].squeeze().to_numpy(),
        "volume": data["Volume"].squeeze().to_numpy(),
    })
    # drop this ticker's previous rows so a rerun refreshes rather than duplicates
    cursor.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
    conn.commit()
    price_rows.to_sql("prices", conn, if_exists="append", index=False)

    # keep the single-column DataFrame just for the log-returns dump below, but
    # store the squeezed Series so the rest of the script works with a plain
    # Series and the .iloc[-1].iloc[0] double-indexing chains disappear.
    closes = data["Close"]
    log_returns = np.log(closes / closes.shift(1))
    print(log_returns)
    all_returns[ticker] = log_returns.squeeze()
    all_closes[ticker] = closes.squeeze()

# Multiplier calculations
returns_df = pd.DataFrame(all_returns) # converting the dictionary to DataFrame
correlation_matrix = returns_df.corr() # calculating the correlation between the columns of the DataFrame
print(correlation_matrix)

avg_correlation = {}
for ticker in tickers:
    row = correlation_matrix[ticker]
    avg_correlation[ticker] = row.drop(ticker).mean()
print(f" average correlation is {avg_correlation}")
multiplier = {}
for ticker in tickers:
    multiplier[ticker] = 1 - avg_correlation[ticker]
print(multiplier)

# Backtest position sizing must not see future correlations: the multiplier above
# is built on the full 10-year sample, which includes the very years being tested.
# For each backtest year, rebuild it from only that year's two-year training window
# (the same window the drift and moving averages train on). Keep this range in
# sync with the backtest loop below.
# every complete year with a full 2-year training window behind it, given
# DATA_START above: first testable year is 2018, last complete year is 2025.
BACKTEST_YEARS = range(2018, 2026)
backtest_multiplier = {}
for year in BACKTEST_YEARS:
    window_corr = returns_df[f"{year-2}-01-01":f"{year}-01-01"].corr()
    backtest_multiplier[year] = {
        t: 1 - window_corr[t].drop(t).mean() for t in tickers
    }

# collecting data from "SPY" for golden cross and evil cross for better probability estimation
spy_data = load_prices("SPY")
spy_closes = spy_data["Close"].squeeze()

# set dictionaries of probability things for the visualizer
S0 = {}
median = {}
p5 = {}
p25 = {}
p75 = {}
p95 = {}

# risk-free rate from the 10-year Treasury yield. It is not ticker-specific, so
# fetch it once here rather than re-fetching the identical value every iteration.
treasury = yf.Ticker("^TNX")
risk_free_rate = treasury.info['previousClose'] / 100

portfolio_pnl = 0
portfolio_invested = 0

##### PASS 2 #####
for i, ticker in enumerate(tickers):

    closes = all_closes[ticker]
    log_returns = all_returns[ticker]
    S0[ticker] = float(closes.iloc[-1])

    mu_annual = log_returns.mean() * 252
    sigma_annual = log_returns.std() * np.sqrt(252)

    print(mu_annual, sigma_annual)

    stock = yf.Ticker(ticker)
    # default to market beta (1.0), not 0: a beta of 0 silently collapses CAPM's
    # expected return to the risk-free rate, which is not a sane fallback.
    beta = stock.info.get('beta', 1)
    if beta is None:
        beta = 1

    market_risk_premium = 0.055

    expected_return = (market_risk_premium * beta) + risk_free_rate

    analyst_target = stock.info.get('targetMeanPrice', None)
    if analyst_target is None:
        implied_return = expected_return # fall back to CAPM if no analyst target
    else:
        implied_return = (analyst_target / S0[ticker]) - 1

    # --- MU & SIGMA ESTIMATION ---
    # drift: blended 20% historical + 20% CAPM + 60% analyst-implied
    mu = float(((mu_annual * 0.2) + (expected_return * 0.2) + (implied_return * 0.6)))
    print(f"Blended mu: {mu:.2%}")
    # volatility
    sigma = sigma_annual

    # --- MONTE CARLO SIMULATION ---
    St = run_monte_carlo(S0[ticker], mu, sigma, T, n, M)
    median[ticker] = np.percentile(St, 50, axis=1)
    p5[ticker] = np.percentile(St, 5, axis=1)
    p25[ticker] = np.percentile(St, 25, axis=1)
    p75[ticker] = np.percentile(St, 75, axis=1)
    p95[ticker] = np.percentile(St, 95, axis=1)

    # define time interval correctly
    time = np.linspace(0,T,n+1)

    final_prices = St[-1]

    # --- PROBABILITY ANALYSIS ---
    # probability that price will be above "X"
    probability = (final_prices > S0[ticker]).mean()
    print(probability)
    ticker_target = S0[ticker] * (1 + PRICE_TARGET_PCT)
    probability2 = (final_prices > ticker_target).mean()
    print(f"Probability of over ${ticker_target:.2f} (+{PRICE_TARGET_PCT:.0%}): {probability2}")
    probability3 = (final_prices > 1.2 * S0[ticker]).mean()
    print(f"Probability of final price over 20% gain: {probability3}")

    # insert fundamentals into database, clearing today's prior row so repeated runs
    # on the same day overwrite instead of piling up duplicates
    cursor.execute(
        "DELETE FROM fundamentals WHERE ticker = ? AND last_updated = ?",
        (ticker, today),
    )
    cursor.execute('''
        INSERT INTO fundamentals
        (ticker, beta, risk_free_rate, analyst_target, expected_return, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, beta, risk_free_rate, analyst_target, expected_return, today))
    conn.commit()

    # --- BACKTEST ---
    result = backtest(ticker, closes, spy_closes, backtest_multiplier,
                      BACKTEST_YEARS, T, n, M, cursor, conn, today)
    portfolio_pnl += result["total_pnl"]
    portfolio_invested += result["total_invested"]

    # Sharpe Ratio Calculation, reported two ways:
    #   traded-years-only  -> "when this strategy trades, how good are those trades"
    #   all-years          -> "return on capital held in this sleeve year-round",
    #                         counting no-trade years as a flat 0%
    # Needs >= 2 samples with some dispersion, else stdev is 0/undefined.
    sharpe_traded = sharpe_ratio(result["traded_returns"], risk_free_rate)
    sharpe_all_years = sharpe_ratio(result["all_year_returns"], risk_free_rate)

    print(f"Sharpe Ratio (traded years only, n={len(result['traded_returns'])}): {sharpe_traded:.2f}")
    print(f"Sharpe Ratio (all years, no-trade=0, n={len(result['all_year_returns'])}): {sharpe_all_years:.2f}")
    print(f"Buy & Hold return ({BACKTEST_YEARS[0]}-{BACKTEST_YEARS[-1]}): {result['p_gain_hold']:.1%}")
    print(f"Backtest accuracy: {result['correct']}/{result['total']} = {result['correct']/result['total']:.0%}")
    print(f"\n--- STRATEGY SUMMARY ---")
    print(f"Total invested: ${result['total_invested']}")
    print(f"Total P&L: ${result['total_pnl']:.2f}")
    print(f"Total return: {(result['total_pnl']/result['total_invested'])*100:.1f}%" if result['total_invested'] > 0 else "No trades")




    # --- CHART ---
    # plot figure
    axes[i].plot(time, median[ticker])
    axes[i].plot(time, p95[ticker])
    axes[i].plot(time, p5[ticker])
    axes[i].fill_between(time, p5[ticker], p95[ticker], alpha=0.3)
    axes[i].fill_between(time, p25[ticker], p75[ticker], alpha=0.4, color='steelblue')
    axes[i].set_xlabel("Years $(t)$")
    axes[i].set_ylabel("Stock Price $S_t$")
    axes[i].set_title(f"{ticker} Stock Price Forecast Across {T} Year(s)")
    axes[i].set_xlim(0, T * 1.15)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=S0[ticker], color='white', linewidth=1, linestyle='--', label=f"Current Price ${S0[ticker]:.2f}")
    axes[i].annotate(f"${p95[ticker][-1]:.1f}", xy=(T, p95[ticker][-1]), color='lime')
    axes[i].annotate(f"${median[ticker][-1]:.1f}", xy=(T, median[ticker][-1]), color='olivedrab')
    axes[i].annotate(f"${p5[ticker][-1]:.1f}", xy=(T, p5[ticker][-1]), color='red')
    axes[i].annotate(f"${p25[ticker][-1]:.1f}", xy=(T, p25[ticker][-1]), color='darkred')
    axes[i].annotate(f"${p75[ticker][-1]:.1f}", xy=(T, p75[ticker][-1]), color='lightgreen')
    axes[i].annotate(f"${S0[ticker]:.1f}", xy=(0, S0[ticker]), color='cyan', ha='right')

    stats_text = f"P(gain): {probability:.0%}\nP(+20%): {probability3:.0%}\nP(${ticker_target:.0f}): {probability2:.0%}"
    axes[i].text(0.02, 0.97, stats_text, transform=axes[i].transAxes,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    axes[i].legend(loc='lower right')

PLV.plot_simulation(tickers, time, median, p5, p25, p75, p95, S0, T)

print(f"\n{'='*40}")
print(f"PORTFOLIO SUMMARY")
print(f"{'='*40}")
print(f"Total invested across all tickers: ${portfolio_invested}")
print(f"Total P&L across all tickers: ${portfolio_pnl:.2f}")
print(f"Total portfolio return: {(portfolio_pnl / portfolio_invested) * 100:.1f}%" if portfolio_invested > 0 else "No trades")

# create heatmap for correlation between tickers
fig2, ax = plt.subplots(figsize=(8,6))
im = ax.imshow(correlation_matrix, cmap='RdYlGn')
# label the Axis for the correlation plot
ax.set_xticks(range(len(tickers)))
ax.set_yticks(range(len(tickers)))
ax.set_xticklabels(tickers)
ax.set_yticklabels(tickers)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
# setting the actual corelation numbers in the corelation heatmap
for row in range(len(tickers)):     # loops through rows 0,1,2,3,4
    for col in range(len (tickers)):    # for each row, loops through columns 0,1,2,3,4
        ax.text(col, row, f"{correlation_matrix.iloc[row, col]:.3f}",
                ha='center', va='center', color='black', fontweight='bold')
ax.set_title("Portfolio Correlation Matrix", pad=20)
plt.colorbar(im, ax=ax)

plt.subplots_adjust(hspace=0.3, wspace=0.4) # manually setting spacing in between graphs

conn.close()

plt.show()
