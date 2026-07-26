import os
import webbrowser

import plotly.graph_objects as go

# traces drawn per ticker (p5, p95-band, p25, p75-band, median); used to build the
# per-ticker visibility masks for the dropdown buttons.
TRACES_PER_TICKER = 5


def _add_band_traces(fig, time, median, p5, p25, p75, p95, name_prefix="", visible=True):
    """Add the five traces that make up one fan chart: the two percentile bands and
    the median line. Shared by the multi-ticker HTML chart and the single-ticker
    dashboard chart so the styling only exists in one place.

    Percentile bands use fill="tonexty": each band trace fills down to the trace
    added just before it, so p95 fills to p5 (5-95% band) and p75 fills to p25
    (25-75% band). The lower edge of each band carries no fill.
    """
    prefix = f"{name_prefix} " if name_prefix else ""
    fig.add_trace(go.Scatter(
        x=time, y=p5, name=f"{prefix}5th pct", visible=visible,
        line=dict(color="rgba(239,85,59,0.6)"),
        hovertemplate="Year %{x:.2f}<br>5th pct: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=time, y=p95, name=f"{prefix}5-95% band", visible=visible,
        line=dict(color="rgba(239,85,59,0.6)"),
        fill="tonexty", fillcolor="rgba(99,110,250,0.12)",
        hovertemplate="Year %{x:.2f}<br>95th pct: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=time, y=p25, name=f"{prefix}25th pct", visible=visible,
        line=dict(color="rgba(99,110,250,0.5)"),
        hovertemplate="Year %{x:.2f}<br>25th pct: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=time, y=p75, name=f"{prefix}25-75% band", visible=visible,
        line=dict(color="rgba(99,110,250,0.5)"),
        fill="tonexty", fillcolor="rgba(70,130,180,0.30)",
        hovertemplate="Year %{x:.2f}<br>75th pct: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=time, y=median, name=f"{prefix}median", visible=visible,
        line=dict(color="white", width=2),
        hovertemplate="Year %{x:.2f}<br>median: $%{y:.2f}<extra></extra>"))


def _s0_line(y):
    """A horizontal reference line at the current price, as a layout shape (so the
    dropdown buttons can swap it per ticker)."""
    return dict(type="line", xref="paper", x0=0, x1=1, yref="y", y0=y, y1=y,
                line=dict(color="cyan", width=1, dash="dash"))


def _prob_annotations(s0, p_gain, p_20):
    """The S0 price label and the P(gain)/P(+20%) box."""
    return [
        dict(xref="paper", x=1.0, xanchor="left", yref="y", y=s0, showarrow=False,
             text=f"S0 ${s0:.2f}", font=dict(color="cyan")),
        dict(xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
             showarrow=False, align="left",
             text=f"P(gain): {p_gain:.0%}<br>P(+20%): {p_20:.0%}",
             bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1,
             font=dict(color="white")),
    ]


def _annotations(ticker, all_S0, prob_gain, prob_20):
    """Per-ticker wrapper around _prob_annotations for the dropdown callbacks."""
    return _prob_annotations(all_S0[ticker], prob_gain[ticker], prob_20[ticker])


def build_fan_chart(ticker, time, median, p5, p25, p75, p95, S0, p_gain, p_20, T):
    """Single-ticker fan chart as a Figure, for embedding (e.g. in Streamlit).
    Same visual language as the multi-ticker HTML chart, minus the dropdown."""
    fig = go.Figure()
    _add_band_traces(fig, time, median, p5, p25, p75, p95)
    fig.add_hline(y=S0, line=dict(color="cyan", width=1, dash="dash"))
    fig.update_layout(
        title=f"{ticker} - Monte Carlo Price Simulation ({T}yr)",
        xaxis_title="Years",
        yaxis_title="Stock Price ($)",
        annotations=_prob_annotations(S0, p_gain, p_20),
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=60, r=90, t=60, b=50),
    )
    return fig


def plot_simulation(tickers, time, all_medians, all_p5, all_p25, all_p75, all_p95,
                    all_S0, prob_gain, prob_20, T):
    fig = go.Figure()

    for i, ticker in enumerate(tickers):
        _add_band_traces(fig, time, all_medians[ticker], all_p5[ticker],
                         all_p25[ticker], all_p75[ticker], all_p95[ticker],
                         name_prefix=ticker, visible=(i == 0))

    # current-price reference line for the initially-visible ticker; the buttons
    # swap it (and the title/annotations) as the selection changes.
    fig.add_hline(y=all_S0[tickers[0]], line=dict(color="cyan", width=1, dash="dash"))

    buttons = []
    for i, ticker in enumerate(tickers):
        visible = [j // TRACES_PER_TICKER == i for j in range(len(tickers) * TRACES_PER_TICKER)]
        buttons.append(dict(
            label=ticker,
            method="update",
            args=[
                {"visible": visible},
                {"title.text": f"Monte Carlo Stock Price Simulation - {ticker} ({T}yr)",
                 "shapes": [_s0_line(all_S0[ticker])],
                 "annotations": _annotations(ticker, all_S0, prob_gain, prob_20)},
            ],
        ))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, font=dict(color="black"),
                          bgcolor="lightblue", bordercolor="white", borderwidth=2,
                          x=1.02, xanchor="left", y=1, yanchor="top")],
        title=f"Monte Carlo Stock Price Simulation - {tickers[0]} ({T}yr)",
        xaxis_title="Years",
        yaxis_title="Stock Price ($)",
        annotations=_annotations(tickers[0], all_S0, prob_gain, prob_20),
        hovermode="x unified",
        template="plotly_dark",
    )

    path = os.path.abspath("output_chart.html")
    fig.write_html(path)
    webbrowser.open(f"file://{path}")
