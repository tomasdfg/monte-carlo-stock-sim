import os
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime

# --- SETTINGS ---
ticker = "AAPL"
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
beta = stock.info['beta']


treasury = yf.Ticker("^TNX")
risk_free_rate = treasury.info['previousClose'] / 100

market_risk_premium = 0.055

expected_return = (market_risk_premium * beta) + risk_free_rate

analyst_target = stock.info['targetMeanPrice']
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
    #check what actually happened:
    actual_return = actual_data.iloc[-1] - actual_data.iloc[0]
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
        (ticker, year, predicted_direction, actual_direction, probability_t, correct, today)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, year, predicted_direction, actual_direction, probability_t, is_correct, today))
    conn.commit()
print(f"Backtest accuracy: {correct}/{total} = {correct/total:.0%}")
    

# --- CHART ---
# plot figure
plt.style.use("dark_background")
plt.plot(time, median)
plt.plot(time, p95)
plt.plot(time, p5)
plt.fill_between(time, p5, p95, alpha=0.3)
plt.fill_between(time, p25, p75, alpha=0.4, color='steelblue')
plt.xlabel("Years $(t)$")
plt.ylabel("Stock Price $S_t$")
plt.title(f"{ticker} Stock Price Forecast Across {T} Year(s)")
plt.xlim(0, T)
plt.grid(True, alpha=0.3)
plt.axhline(y=S0, color='white', linewidth=1, linestyle='--', label=f"Current Price ${S0:.2f}")
plt.annotate(f"${p95[-1]:.1f}", xy=(T, p95[-1]), color='lime')
plt.annotate(f"${median[-1]:.1f}", xy=(T, median[-1]), color='olivedrab')
plt.annotate(f"${p5[-1]:.1f}", xy=(T, p5[-1]), color='red')
plt.annotate(f"${p25[-1]:.1f}", xy=(T, p25[-1]), color='darkred')
plt.annotate(f"${p75[-1]:.1f}", xy=(T, p75[-1]), color='lightgreen')
plt.annotate(f"${S0:.1f}", xy=(0, S0), color='cyan', ha='right')

stats_text = f"P(gain): {probability:.0%}\nP(+20%): {probability3:.0%}\nP(${price_target}): {probability2:.0%}"
plt.text(0.02, 0.97, stats_text, transform=plt.gca().transAxes,
         verticalalignment='top', color='white',
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
plt.legend(loc='lower right')
plt.show()
