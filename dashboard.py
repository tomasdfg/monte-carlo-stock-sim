"""Streamlit dashboard for the Monte Carlo stock simulator.

Run with:  streamlit run dashboard.py

Everything here drives gbm_core and plotly_viz rather than reimplementing the
model, so the dashboard, the chart script and the robustness sweep always agree.

Note on the sliders: the drift-blend weights change the FORWARD projection only.
The backtest deliberately does not use them - it was de-leaked in Phase 2 (today's
analyst target and CAPM inputs are unknowable at a 2013 backtest date) and since
Phase 5.2 it bootstraps each training window's own daily returns. That separation
is a feature, and the Backtest tab says so on screen.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from gbm_core import (load_prices, run_monte_carlo, backtest, backtest_years,
                      train_start_year, sharpe_ratio, win_rate, max_drawdown,
                      simulate_correlated_portfolio, TRANSACTION_COST,
                      TRAINING_YEARS, WINDOW_MODE)
from plotly_viz import build_fan_chart

TICKERS = ["AAPL", "NVDA", "AMZN", "MSFT", "GLD"]
DB_PATH = os.path.join("data", "market_data.db")
MARKET_RISK_PREMIUM = 0.055

st.set_page_config(page_title="Monte Carlo Stock Dashboard", page_icon="📈",
                   layout="wide")


# --------------------------------------------------------------------------
# data loading (cached so moving a slider does not re-download anything)
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading price history...")
def load_closes(ticker):
    """Daily closes as a plain Series, NaN rows dropped so the last close is real."""
    return load_prices(ticker)["Close"].dropna().squeeze()


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker):
    """Beta and analyst target from yfinance. Returns sane fallbacks (and a flag)
    if the lookup fails, so the dashboard still works offline."""
    try:
        info = yf.Ticker(ticker).info
        beta = info.get("beta", 1)
        if beta is None:
            beta = 1
        return float(beta), info.get("targetMeanPrice", None), True
    except Exception:
        return 1.0, None, False


@st.cache_data(ttl=3600, show_spinner=False)
def load_risk_free_rate():
    try:
        return yf.Ticker("^TNX").info["previousClose"] / 100, True
    except Exception:
        return 0.04, False


@st.cache_data(show_spinner="Preparing backtest inputs...")
def prepare_backtest_inputs():
    """Closes for every ticker plus SPY, the walk-forward span, and the per-year
    training-window correlation multiplier used for position sizing."""
    closes = {t: load_closes(t) for t in TICKERS}
    returns_df = pd.DataFrame(
        {t: np.log(closes[t] / closes[t].shift(1)) for t in TICKERS})
    years = backtest_years(pd.Timestamp(closes[TICKERS[0]].index[-1]))
    multiplier = {}
    for year in years:
        ts = train_start_year(year)
        window_corr = returns_df[f"{ts}-01-01":f"{year}-01-01"].corr()
        multiplier[year] = {t: 1 - window_corr[t].drop(t).mean() for t in TICKERS}
    return closes, load_closes("SPY"), multiplier, list(years), returns_df


@st.cache_data(show_spinner="Running the walk-forward backtest for every ticker...")
def run_all_backtests(seed, M, n, T):
    """Backtest every ticker under one seed and return the per-year detail plus a
    per-ticker summary. Cheap (well under a second for all five), so the accuracy
    grid can be computed live instead of depending on a database that a fresh
    deploy does not have.

    Consumes one throwaway forward-simulation draw per ticker so the RNG stream
    lines up with the chart script, which simulates then backtests each ticker in
    turn - that keeps these numbers equal to the stored/committed run.
    """
    all_closes, spy_closes, multiplier, years, _ = prepare_backtest_inputs()
    np.random.seed(int(seed))
    detail, summary = [], {}
    for t in TICKERS:
        np.random.normal(0, np.sqrt(T / n), size=(M, n))
        res = backtest(t, all_closes[t], spy_closes, multiplier, years, M,
                       verbose=False)
        detail.extend(res["yearly_detail"])
        summary[t] = res
    return pd.DataFrame(detail), summary, years


@st.cache_data(show_spinner=False)
def read_backtest_results():
    """Stored backtest rows, read-only so the dashboard can never corrupt the DB
    that the chart script writes."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            return pd.read_sql_query(
                "SELECT ticker, year, predicted_direction, actual_direction,"
                " probability_t, is_correct, today FROM backtest_results", conn)
    except Exception:
        return None


# --------------------------------------------------------------------------
# sidebar controls
# --------------------------------------------------------------------------

st.sidebar.title("Controls")
ticker = st.sidebar.selectbox("Ticker", TICKERS, index=0)

st.sidebar.subheader("Simulation")
T = st.sidebar.slider("Horizon T (years)", 0.25, 5.0, 1.0, 0.25)
M = st.sidebar.slider("Simulations M", 200, 10000, 1000, 200)
n = st.sidebar.slider("Steps n", 50, 500, 100, 50)
seed = st.sidebar.number_input("Random seed", value=42, step=1,
                               help="Same seed gives the same paths, so slider "
                                    "changes show real effects, not noise.")

st.sidebar.subheader("Drift blend weights")
st.sidebar.caption("Forward projection only - the backtest is de-leaked and "
                   "ignores these.")
w_hist = st.sidebar.slider("Historical", 0.0, 1.0, 0.2, 0.05)
w_capm = st.sidebar.slider("CAPM", 0.0, 1.0, 0.2, 0.05)
w_analyst = st.sidebar.slider("Analyst target", 0.0, 1.0, 0.6, 0.05)
weight_total = w_hist + w_capm + w_analyst
if weight_total == 0:
    w_hist = w_capm = w_analyst = 1 / 3          # degenerate input, fall back to equal
    weight_total = 1.0
nw_hist, nw_capm, nw_analyst = (w_hist / weight_total, w_capm / weight_total,
                                w_analyst / weight_total)
if abs(weight_total - 1.0) > 1e-9:
    st.sidebar.info(f"Weights sum to {weight_total:.2f}; normalized to "
                    f"{nw_hist:.2f}/{nw_capm:.2f}/{nw_analyst:.2f}.")

target_pct = st.sidebar.slider("Price target (% above spot)", 0.05, 1.0, 0.30, 0.05)


# --------------------------------------------------------------------------
# forward projection
# --------------------------------------------------------------------------

closes = load_closes(ticker)
log_returns = np.log(closes / closes.shift(1))
S0 = float(closes.iloc[-1])
hist_mu = float(log_returns.mean() * 252)
sigma = float(log_returns.std() * np.sqrt(252))

beta, analyst_target, fundamentals_ok = load_fundamentals(ticker)
risk_free_rate, rf_ok = load_risk_free_rate()
capm_return = MARKET_RISK_PREMIUM * beta + risk_free_rate
implied_return = (analyst_target / S0 - 1) if analyst_target else capm_return

mu = nw_hist * hist_mu + nw_capm * capm_return + nw_analyst * implied_return

np.random.seed(int(seed))
St = run_monte_carlo(S0, mu, sigma, T, n, M)
time = np.linspace(0, T, n + 1)
median = np.percentile(St, 50, axis=1)
p5, p25 = np.percentile(St, 5, axis=1), np.percentile(St, 25, axis=1)
p75, p95 = np.percentile(St, 75, axis=1), np.percentile(St, 95, axis=1)
final_prices = St[-1]

p_gain = float((final_prices > S0).mean())
p_20 = float((final_prices > 1.2 * S0).mean())
p_target = float((final_prices > S0 * (1 + target_pct)).mean())

st.title("Monte Carlo Stock Dashboard")
st.caption(f"{ticker} - spot ${S0:,.2f} - history {closes.index[0]} to "
           f"{closes.index[-1]} - re-simulated live on every control change")
if not (fundamentals_ok and rf_ok):
    st.warning("Could not reach yfinance for fundamentals/treasury; using "
               "fallbacks (beta 1.0, risk-free 4%).")

forecast_tab, backtest_tab, portfolio_tab = st.tabs(
    ["Forecast", "Backtest", "Portfolio risk"])

with forecast_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot", f"${S0:,.2f}")
    c2.metric("Blended drift", f"{mu:.1%}")
    c3.metric("Volatility", f"{sigma:.1%}")
    c4.metric("Median outcome", f"${median[-1]:,.2f}",
              delta=f"{median[-1] / S0 - 1:.1%}")
    c5.metric("P(gain)", f"{p_gain:.0%}")

    st.plotly_chart(
        build_fan_chart(ticker, time, median, p5, p25, p75, p95, S0, p_gain, p_20, T),
        width='stretch')

    left, right = st.columns(2)
    with left:
        st.subheader("Outcome probabilities")
        st.dataframe(pd.DataFrame({
            "Outcome": ["Any gain", "Gain > 20%",
                        f"Above target (+{target_pct:.0%}, ${S0 * (1 + target_pct):,.2f})",
                        "Any loss"],
            "Probability": [f"{p_gain:.1%}", f"{p_20:.1%}", f"{p_target:.1%}",
                            f"{1 - p_gain:.1%}"],
        }), hide_index=True, width='stretch')
        st.caption(f"5th pct ${np.percentile(final_prices, 5):,.2f} - "
                   f"95th pct ${np.percentile(final_prices, 95):,.2f} at T={T}yr")
    with right:
        st.subheader("Drift blend")
        st.dataframe(pd.DataFrame({
            "Component": ["Historical", f"CAPM (beta {beta:.2f})",
                          ("Analyst target $%.2f" % analyst_target) if analyst_target
                          else "Analyst (n/a, using CAPM)"],
            "Annual return": [f"{hist_mu:.1%}", f"{capm_return:.1%}",
                              f"{implied_return:.1%}"],
            "Weight": [f"{nw_hist:.0%}", f"{nw_capm:.0%}", f"{nw_analyst:.0%}"],
        }), hide_index=True, width='stretch')
        st.caption(f"Blended drift {mu:.2%} - risk-free {risk_free_rate:.2%}")

with backtest_tab:
    st.info(f"Walk-forward: {WINDOW_MODE} {TRAINING_YEARS}-year training window, "
            f"bootstrapped daily returns, {TRANSACTION_COST:.1%} cost per trade. "
            "The drift-blend sliders intentionally do not apply here - the backtest "
            "may only use information available at each historical date.")

    # --- live accuracy grid: computed on the fly for every ticker, so it is here
    # whether or not a SQLite database exists (a fresh deploy has none) ---
    detail_df, summary, bt_years = run_all_backtests(int(seed), M, n, T)

    st.subheader("Accuracy by ticker and year")
    st.caption(f"Green = the direction call for that year was right, red = wrong. "
               f"Computed live for seed {int(seed)}, M={M}, {bt_years[0]}-{bt_years[-1]}.")
    grid = detail_df.pivot(index="ticker", columns="year", values="correct")
    st.dataframe(grid.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1)
                 .format("{:.0f}"), width='stretch')

    hits, cells = int(detail_df["correct"].sum()), len(detail_df)
    g1, g2, g3 = st.columns(3)
    g1.metric("Overall accuracy", f"{hits}/{cells}", delta=f"{hits / cells:.0%}")
    worst_year = detail_df.groupby("year")["correct"].mean().idxmin()
    g2.metric("Worst year", str(worst_year),
              delta=f"{detail_df[detail_df['year'] == worst_year]['correct'].mean():.0%}",
              delta_color="inverse")
    g3.metric("Trades placed", int(detail_df["traded"].sum()))

    st.subheader("Per-ticker summary")
    st.dataframe(pd.DataFrame([{
        "ticker": t,
        "accuracy": f"{summary[t]['correct']}/{summary[t]['total']}",
        "return": (f"{summary[t]['total_pnl'] / summary[t]['total_invested']:.1%}"
                   if summary[t]["total_invested"] else "n/a"),
        "win rate": ("n/a" if win_rate(summary[t]["traded_returns"]) is None
                     else f"{win_rate(summary[t]['traded_returns']):.0%}"),
        "max drawdown": f"{max_drawdown(summary[t]['all_year_returns']):.1%}",
        "sharpe (traded)": f"{sharpe_ratio(summary[t]['traded_returns'], risk_free_rate):.2f}",
        "buy & hold": f"{summary[t]['p_gain_hold']:.0%}",
    } for t in TICKERS]), hide_index=True, width='stretch')

    with st.expander(f"Year-by-year detail for {ticker}"):
        st.dataframe(
            detail_df[detail_df["ticker"] == ticker]
            .assign(probability=lambda d: d["probability"].map("{:.1%}".format))
            .drop(columns=["ticker"]),
            hide_index=True, width='stretch')

    # --- stored results, when a database happens to be present ---
    stored = read_backtest_results()
    st.subheader("Stored results (SQLite)")
    if stored is None or stored.empty:
        st.info(
            "No database found - data/market_data.db is written by "
            "`python3 Brownian_Motion_1st_Iteration.py` and is gitignored, so a "
            "fresh deploy starts without it. Everything above is computed live and "
            "needs no database.")
    else:
        runs = sorted(stored["today"].unique(), reverse=True)
        chosen_run = st.selectbox("Run date", runs, index=0)
        run_rows = stored[stored["today"] == chosen_run]
        only_this = st.checkbox(f"Show {ticker} only", value=True)
        shown = run_rows[run_rows["ticker"] == ticker] if only_this else run_rows

        a, b, c = st.columns(3)
        a.metric("Rows", len(shown))
        if len(shown):
            a_acc = shown["is_correct"].mean()
            b.metric("Accuracy", f"{shown['is_correct'].sum()}/{len(shown)}",
                     delta=f"{a_acc:.0%}")
            c.metric("Mean P(gain)", f"{shown['probability_t'].mean():.0%}")
        st.dataframe(
            shown.assign(probability_t=shown["probability_t"].map("{:.1%}".format))
                 .rename(columns={"probability_t": "P(gain)",
                                  "predicted_direction": "predicted",
                                  "actual_direction": "actual",
                                  "is_correct": "correct"}),
            hide_index=True, width='stretch')
        st.caption("These are rows the chart script previously wrote. The grid at "
                   "the top of this tab is the live equivalent.")

    st.subheader("Re-run the backtest live")
    st.caption("Uses the same gbm_core.backtest the chart script and robustness "
               "sweep call - so numbers here match those runs for a given seed.")
    if st.button(f"Run walk-forward backtest for {ticker}"):
        all_closes, spy_closes, multiplier, years, _ = prepare_backtest_inputs()
        with st.spinner(f"Backtesting {ticker} over {years[0]}-{years[-1]}..."):
            np.random.seed(int(seed))
            result = backtest(ticker, all_closes[ticker], spy_closes, multiplier,
                              years, M, verbose=False)
        # keep the run in session state, otherwise the next widget interaction
        # reruns the script, the button reads False, and the results vanish
        st.session_state["live_backtest"] = (ticker, int(seed), M, result, list(years))

    if "live_backtest" in st.session_state:
        bt_ticker, bt_seed, bt_M, result, years = st.session_state["live_backtest"]
        st.caption(f"Showing live run: {bt_ticker}, seed {bt_seed}, M={bt_M}, "
                   f"{years[0]}-{years[-1]}"
                   + ("" if bt_ticker == ticker
                      else f"  -  (sidebar now shows {ticker}; re-run to update)"))
        traded = result["traded_returns"]
        wr = win_rate(traded)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['correct']}/{result['total']}",
                  delta=f"{result['correct'] / result['total']:.0%}")
        m2.metric("Return on capital",
                  f"{result['total_pnl'] / result['total_invested']:.1%}"
                  if result["total_invested"] else "n/a")
        m3.metric("Win rate", "n/a" if wr is None else f"{wr:.0%}",
                  delta=f"{len(traded)} trades")
        m4.metric("Max drawdown", f"{max_drawdown(result['all_year_returns']):.1%}")
        d1, d2, d3 = st.columns(3)
        d1.metric("Sharpe (traded)",
                  f"{sharpe_ratio(traded, risk_free_rate):.2f}")
        d2.metric("Buy & hold", f"{result['p_gain_hold']:.0%}")
        d3.metric("Costs paid", f"${result['total_cost']:,.2f}")
        st.dataframe(pd.DataFrame({
            "Year": years,
            "Return": [f"{r:.1%}" for r in result["all_year_returns"]],
            "P&L": [f"${p:,.2f}" for p in result["yearly_pnl"]],
            "Invested": [f"${i:,.2f}" for i in result["yearly_invested"]],
        }), hide_index=True, width='stretch')

with portfolio_tab:
    st.subheader("Correlated portfolio simulation (Cholesky)")
    st.caption("Simulates all five names jointly using the historical correlation "
               "matrix, then compares against an independent simulation to show "
               "how much diversification the independence assumption invents.")
    _, _, _, _, returns_df = prepare_backtest_inputs()
    corr = returns_df.corr().loc[TICKERS, TICKERS]
    mu_vec = np.array([returns_df[t].mean() * 252 for t in TICKERS])
    sigma_vec = np.array([returns_df[t].std() * np.sqrt(252) for t in TICKERS])
    weights = np.full(len(TICKERS), 1 / len(TICKERS))

    np.random.seed(int(seed))
    correlated = simulate_correlated_portfolio(mu_vec, sigma_vec, corr.to_numpy(),
                                               weights, T, n, M) - 1.0
    independent = simulate_correlated_portfolio(mu_vec, sigma_vec,
                                                np.eye(len(TICKERS)), weights,
                                                T, n, M) - 1.0
    var5 = -np.percentile(correlated, 5)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected return", f"{correlated.mean():.1%}")
    k2.metric("5% VaR", f"{var5:.1%}")
    k3.metric("P(loss)", f"{(correlated < 0).mean():.0%}")
    k4.metric("Volatility", f"{correlated.std():.1%}",
              delta=f"{correlated.std() - independent.std():+.1%} vs independent")

    import plotly.graph_objects as go
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=correlated * 100, nbinsx=60, name="correlated",
                                marker_color="steelblue"))
    hist.add_trace(go.Histogram(x=independent * 100, nbinsx=60, name="independent",
                                marker_color="darkorange", opacity=0.45))
    hist.add_vline(x=-var5 * 100, line=dict(color="red", dash="dot"),
                   annotation_text=f"5% VaR {var5:.0%}")
    hist.update_layout(barmode="overlay", template="plotly_dark",
                       title=f"Equal-weight portfolio return distribution ({T}yr)",
                       xaxis_title="Portfolio return (%)", yaxis_title="Frequency")
    st.plotly_chart(hist, width='stretch')

    st.markdown(
        f"Correlation removes **{1 - independent.std() / correlated.std():.0%}** of "
        "the diversification an independent simulation would assume - these names "
        "fall together.")
    st.dataframe(corr.style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=1)
                 .format("{:.3f}"), width='stretch')
