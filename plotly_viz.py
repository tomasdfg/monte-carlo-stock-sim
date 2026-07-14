import os
import webbrowser

import plotly.graph_objects as go
import numpy as np


def plot_simulation(tickers, time, all_medians, all_p5, all_p25, all_p75, all_p95, all_S0, T):
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        print(i, i == 0)
        fig.add_trace(go.Scatter(
            x=time,
            y=all_medians[ticker],
            name=f"{ticker} Median",
            visible= (i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=all_p5[ticker],
            name=f"{ticker} p5",
            visible= (i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=all_p25[ticker],
            name=f"{ticker} p25",
            visible= (i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=all_p75[ticker],
            name=f"{ticker} p75",
            visible= (i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=all_p95[ticker],
            name=f"{ticker} p95",
            visible= (i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=time,
            y=np.full(len(time), all_S0[ticker]),
            name=f"{ticker} S0",
            visible= (i == 0)
        ))
    buttons = []
    for i, ticker in enumerate(tickers):
        buttons.append(dict(
        label=ticker,
        method="update",
        args=[{"visible": [j // 6 == i for j in range(len(tickers) * 6)]}]
        ))
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            font=dict(color="black"),
            bgcolor="blue",
            bordercolor="white",
            borderwidth=2,
        )],
        title="Monte Carlo Stock Price Simulation",
        xaxis_title="Years",
        yaxis_title="Stock Price",
        template="plotly_dark"    
    )

    path = os.path.abspath("output_chart.html")
    fig.write_html(path)
    webbrowser.open(f"file://{path}")