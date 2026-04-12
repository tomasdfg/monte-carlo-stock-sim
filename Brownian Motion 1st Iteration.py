import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

# --- SETTINGS ---
tickers = ["AAPL", "NVDA", "AMZN", "MSFT", "GLD"]
# number of steps
n = 100
# time in years
T = 1
# number of sims
M = 1000
# target price
price_target = 300

# --- DATA LOADING ---
today = datetime.today().strftime('%Y-%m-%d')
# create data folder
os.makedirs("data", exist_ok=True)

# create database and tables
conn = sqlite3.connect("data/market_data.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT,
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER
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

plt.style.use("dark_background") # sets background to black
fig, axes = plt.subplots(2, 3, figsize=(20,12))
axes = axes.flatten() # converts 2D grid to a simple list
plt.style.use("dark_background")

for i, ticker in enumerate(tickers):

    if os.path.exists(f"data/{ticker}_data.csv"):
        file_age = datetime.today() - datetime.fromtimestamp(os.path.getmtime(f"data/{ticker}_data.csv"))
    else:
        file_age = None

    if file_age is not None and file_age.days < 1:
        data = pd.read_csv(f"data/{ticker}_data.csv", index_col=0, header=[0,1])
    else:
        data = yf.download(ticker, period="10y", auto_adjust=True)
        data.to_csv(f"data/{ticker}_data.csv")



    # save downloaded price history for {ticker} to the prices table in the database
    data.to_sql("prices", conn, if_exists="replace", index=True)

    closes = data["Close"]
    print(closes)

    # initial stock price
    S0 = float(closes.iloc[-1].iloc[0])

    log_returns = np.log(closes / closes.shift(1))
    print(log_returns)

    mu_calc = log_returns.mean() 
    sigma_calc = log_returns.std() 
    mu_annual = mu_calc * 252
    sigma_annual = sigma_calc * np.sqrt(252)

    print(mu_annual, sigma_annual)

    stock = yf.Ticker(ticker)
    beta = stock.info.get('beta', 0)
    if beta is None:
        beta = 0


    treasury = yf.Ticker("^TNX")
    risk_free_rate = treasury.info['previousClose'] / 100

    market_risk_premium = 0.055

    expected_return = (market_risk_premium * beta) + risk_free_rate

    analyst_target = stock.info.get('targetMeanPrice', None)
    if analyst_target is None:
        implied_return = expected_return # fall back to CAPM if no analyst target
    else:
        implied_return = (analyst_target / S0) - 1

    # --- MU & SIGMA ESTIMATION ---
    # drift: blended 40% historical + 60% CAPM
    mu = float(((mu_annual * 0.2) + (expected_return * 0.2) + (implied_return * 0.4)).values[0])
    print(f"Blended mu: {mu:.2%}")
    # volatility
    sigma = sigma_annual.values[0]

    # --- MONTE CARLO SIMULATION ---
    # calc each time step
    dt = T/n
    # simulation using numpy arrays
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(M), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0)
    St = S0 * St.cumprod(axis=0)
    median = np.percentile(St, 50, axis=1)
    p5 = np.percentile(St, 5, axis=1)
    p95 = np.percentile(St, 95, axis=1)
    p25 = np.percentile(St, 25, axis=1)
    p75 = np.percentile(St, 75, axis=1)
    # define time interval correctly
    time = np.linspace(0,T,n+1)


    final_prices = St[-1]

    # --- PROBABILITY ANALYSIS ---
    # probability that price will be above "X"
    probability = (final_prices > S0).mean()
    print(probability)
    probability2 = (final_prices > price_target).mean()
    print(f"Probability of over ${price_target}: {probability2}")
    probability3 = (final_prices > 1.2 * S0).mean()
    print(f"Probability of final price over 20% gain: {probability3}")

    # insert fundamentals into database
    cursor.execute('''
        INSERT INTO fundamentals 
        (ticker, beta, risk_free_rate, analyst_target, expected_return, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, beta, risk_free_rate, analyst_target, expected_return, today))
    conn.commit()

    


    correct = 0
    total = 0
    total_pnl = 0
    total_invested = 0
    yearly_returns = []

    for year in range(2018, 2024):
        training_data = closes[f"{year-2}-01-01":f"{year}-01-01"]
        actual_data = closes[f"{year}-01-01":f"{year+1}-01-01"]
        #calc MU and SIGMA from T-data:
        log_returns_t = np.log(training_data / training_data.shift(1))
        mu_calc_t = log_returns_t.mean() 
        sigma_calc_t = log_returns_t.std() 
        mu_annual_t = mu_calc_t * 252
        sigma_annual_t = sigma_calc_t * np.sqrt(252)
        # drift: blended 40% historical + 60% CAPM
        mu_t = float(((mu_annual_t * 0.2) + (expected_return * 0.2) + (implied_return * 0.4)).values[0])
        # volatility
        sigma_t = sigma_annual_t.values[0]
        # calculate Moving Averages and Signal MA
        ma_50_t = training_data.rolling(50).mean()
        ma_200_t = training_data.rolling(200).mean()
        ma_signal_t = "BUY" if ma_50_t.iloc[-1].iloc[0] > ma_200_t.iloc[-1].iloc[0] else "AVOID"
        # get starting price for year:
        Starting_Price_t = float(actual_data.iloc[0].iloc[0])
        # run Monte Carlo Sim:
        # calc each time step
        dt_t = T/n
        # simulation using numpy arrays
        St_t = np.exp(
            (mu_t - sigma_t ** 2 / 2) * dt_t
            + sigma_t * np.random.normal(0, np.sqrt(dt_t), size=(M,n)).T
        )
        # include array of 1's
        St_t = np.vstack([np.ones(M), St_t])
        # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0)
        St_t = Starting_Price_t * St_t.cumprod(axis=0)
        median_t = np.percentile(St_t, 50, axis=1)
        p5_t = np.percentile(St_t, 5, axis=1)
        p95_t = np.percentile(St_t, 95, axis=1)
        p25_t = np.percentile(St_t, 25, axis=1)
        p75_t = np.percentile(St_t, 75, axis=1)
        # define time interval correctly
        time_t = np.linspace(0,T,n+1)
        final_prices_t = St_t[-1]
        # probability that price will be above "X"
        probability_t = (final_prices_t > Starting_Price_t).mean()
        probability3_t = (final_prices_t > 1.2 * Starting_Price_t).mean()
        #how much is put in for each trade
        investment = position_size(probability_t)
        # strength confidence signal
        print(f"{year}: P={probability_t:.0%}, MA={ma_signal_t}, Investment=%{investment}")
        actual_return = actual_data.iloc[-1] - actual_data.iloc[0]
        if (probability_t > 0.63) and (ma_signal_t == 'BUY'):
            print(f"{year}: Strong Buy")
            # calculate take profit
            take_profit_hit = np.any(actual_data > (Starting_Price_t * 1.2))
            # calculate profit/loss in dollars
            if take_profit_hit:
                profit_loss = investment * 0.20
                yearly_returns.append(0.20)
            else:
                profit_loss = investment * (actual_return.iloc[0] / Starting_Price_t)
                yearly_returns.append(0)
            total_pnl += profit_loss
            total_invested += investment
            print(f"{year}: Investment=${investment}, P&L=${profit_loss:.2f}")
            print(f"Take profit hit: {take_profit_hit}")
        else:
            print(f"{year}: No Trade")
            yearly_returns.append(0)

        #check what actually happened:
        predicted_direction = "UP" if probability_t > 0.5 else "DOWN"
        actual_direction = "UP" if actual_return.iloc[0] > 0 else "DOWN"
        is_correct = 1 if predicted_direction == actual_direction else 0
        print(f"{year}: P(gain)={probability_t:.0%}, Actual={'UP' if actual_return.iloc[0] > 0 else 'DOWN'}")
        if (probability_t > 0.5) and (actual_return.iloc[0] > 0):
            correct +=1
            total +=1
        else:
            total +=1
        
        cursor.execute('''
            INSERT INTO backtest_results
            (ticker, year, predicted_direction, actual_direction, probability_t, is_correct, today)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (ticker, year, predicted_direction, actual_direction, probability_t, is_correct, today))
        conn.commit()
    #calculating what the hold amount would be
    bh_start = closes["2018-01-01" : "2018-02-01"].iloc[0].iloc[0]
    bh_end = closes["2023-01-01" : "2024-01-01"].iloc[-1].iloc[0]
    p_gain_hold = (bh_end / bh_start) - 1

    # Sharpe Ratio Calculation
    if np.std(yearly_returns, ddof=1) > 0:
        Sharpe = (np.mean(yearly_returns) - risk_free_rate) / np.std(yearly_returns, ddof=1)
    else:
        Sharpe = 0    

    print(f"Sharpe Ratio: {Sharpe:.2f}")
    print(f"Buy & Hold return (2018-2023): {p_gain_hold:.1%}")
    print(f"Backtest accuracy: {correct}/{total} = {correct/total:.0%}")
    print(f"/n--- STRATEGY SUMMARY ---")
    print(f"Total invested: ${total_invested}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Total return: {(total_pnl/total_invested)*100:.1f}%" if total_invested > 0 else "No trades")





    # --- CHART ---
    # plot figure
    axes[i].plot(time, median)
    axes[i].plot(time, p95)
    axes[i].plot(time, p5)
    axes[i].fill_between(time, p5, p95, alpha=0.3)
    axes[i].fill_between(time, p25, p75, alpha=0.4, color='steelblue')
    axes[i].set_xlabel("Years $(t)$")
    axes[i].set_ylabel("Stock Price $S_t$")
    axes[i].set_title(f"{ticker} Stock Price Forecast Across {T} Year(s)")
    axes[i].set_xlim(0, T * 1.15)
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(y=S0, color='white', linewidth=1, linestyle='--', label=f"Current Price ${S0:.2f}")
    axes[i].annotate(f"${p95[-1]:.1f}", xy=(T, p95[-1]), color='lime')
    axes[i].annotate(f"${median[-1]:.1f}", xy=(T, median[-1]), color='olivedrab')
    axes[i].annotate(f"${p5[-1]:.1f}", xy=(T, p5[-1]), color='red')
    axes[i].annotate(f"${p25[-1]:.1f}", xy=(T, p25[-1]), color='darkred')
    axes[i].annotate(f"${p75[-1]:.1f}", xy=(T, p75[-1]), color='lightgreen')
    axes[i].annotate(f"${S0:.1f}", xy=(0, S0), color='cyan', ha='right')

    stats_text = f"P(gain): {probability:.0%}\nP(+20%): {probability3:.0%}\nP(${price_target}): {probability2:.0%}"
    axes[i].text(0.02, 0.97, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    axes[i].legend(loc='lower right')
plt.tight_layout() # this auto adjusts spacing so no overlap
plt.show()
