"""
visualizer.py
-------------
Plotly chart library for the Social Sentiment Demand Tracker.
All charts return plotly Figure objects — compatible with both
Streamlit (st.plotly_chart) and Jupyter (fig.show()).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Brand colour palette ──────────────────────────────────────────────────────
COLOURS = {
    "Stanley Tumbler":  "#2196F3",
    "Crocs Classic":    "#FF9800",
    "Ninja Blender":    "#F44336",
    "Oura Ring":        "#9C27B0",
    "Athletic Greens":  "#4CAF50",
}
POS_COLOUR  = "#00C853"
NEG_COLOUR  = "#D50000"
NEUT_COLOUR = "#9E9E9E"
BG_COLOUR   = "#0e1117"
GRID_COLOUR = "#2a2a3e"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=16, color="#ffffff")),
        plot_bgcolor=BG_COLOUR,
        paper_bgcolor=BG_COLOUR,
        font=dict(color="#cccccc", size=11),
        xaxis=dict(gridcolor=GRID_COLOUR, showgrid=True),
        yaxis=dict(gridcolor=GRID_COLOUR, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#444"),
        margin=dict(l=50, r=30, t=60, b=40),
    )


def plot_sdds_timeseries(ddm_series: pd.DataFrame, product: str) -> go.Figure:
    """Hourly SDDS and DDM time series for one product."""
    df = ddm_series[ddm_series["product"] == product].sort_values("hour")
    colour = COLOURS.get(product, "#2196F3")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["SDDS  (hourly sentiment score)", "Sentiment Velocity"],
    )

    # SDDS bars coloured by sign
    bar_colours = [POS_COLOUR if v >= 0 else NEG_COLOUR for v in df["sdds_final"]]
    fig.add_trace(
        go.Bar(
            x=df["hour"], y=df["sdds_final"],
            marker_color=bar_colours, name="SDDS",
            opacity=0.6,
        ),
        row=1, col=1,
    )

    # DDM line
    fig.add_trace(
        go.Scatter(
            x=df["hour"], y=df["ddm"],
            mode="lines", name="DDM",
            line=dict(color=colour, width=2.5),
        ),
        row=1, col=1,
    )

    # Zero reference
    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=1, col=1)

    # Sentiment velocity
    vel_colours = [POS_COLOUR if v >= 0 else NEG_COLOUR for v in df["sentiment_velocity"]]
    fig.add_trace(
        go.Bar(
            x=df["hour"], y=df["sentiment_velocity"],
            marker_color=vel_colours, name="Velocity",
            opacity=0.7, showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#555", row=2, col=1)

    layout = _base_layout(f"{product} — SDDS & DDM Signal")
    layout.update(height=480, showlegend=True)
    fig.update_layout(**layout)
    fig.update_yaxes(range=[-1.1, 1.1], row=1, col=1)
    return fig


def plot_ddm_comparison(latest: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing latest DDM across all products."""
    df = latest.sort_values("DDM")
    bar_colours = [POS_COLOUR if v >= 0 else NEG_COLOUR for v in df["DDM"]]

    fig = go.Figure(
        go.Bar(
            x=df["DDM"],
            y=df["Product"],
            orientation="h",
            marker_color=bar_colours,
            text=[f"{v:+.3f}" for v in df["DDM"]],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#888")
    layout = _base_layout("Latest DDM — All Products")
    layout.update(height=320, xaxis=dict(range=[-1.1, 1.1], gridcolor=GRID_COLOUR))
    fig.update_layout(**layout)
    return fig


def plot_sentiment_heatmap(hourly_sdds: pd.DataFrame) -> go.Figure:
    """
    Heatmap: products (rows) × hours (columns) coloured by SDDS.
    """
    pivot = hourly_sdds.pivot_table(
        index="product", columns="hour", values="sdds_final", aggfunc="mean"
    )
    # Downsample to last 7 days × 6-hour buckets for readability
    pivot.columns = pd.to_datetime(pivot.columns)
    pivot = pivot.resample("6h", axis=1).mean()
    col_labels = [c.strftime("%a %H:%M") for c in pivot.columns]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=col_labels,
            y=list(pivot.index),
            colorscale=[
                [0.0, NEG_COLOUR],
                [0.5, "#1a1a2e"],
                [1.0, POS_COLOUR],
            ],
            zmin=-1, zmax=1,
            colorbar=dict(title="SDDS", tickvals=[-1, 0, 1]),
        )
    )
    layout = _base_layout("Sentiment Heatmap — Products × Time")
    layout.update(height=280)
    fig.update_layout(**layout)
    return fig


def plot_post_volume(scored_posts: pd.DataFrame) -> go.Figure:
    """Stacked area chart of post volume by platform over time."""
    df = scored_posts.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    vol = df.groupby(["hour", "platform"]).size().reset_index(name="count")

    fig = px.area(
        vol, x="hour", y="count", color="platform",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    layout = _base_layout("Post Volume by Platform")
    layout.update(height=300)
    fig.update_layout(**layout)
    return fig


def plot_gauge(ddm_value: float, product: str) -> go.Figure:
    """Gauge chart showing current DDM for a single product."""
    if ddm_value >= 0.6:
        colour = POS_COLOUR
    elif ddm_value >= 0.25:
        colour = "#8BC34A"
    elif ddm_value >= -0.3:
        colour = NEUT_COLOUR
    elif ddm_value >= -0.55:
        colour = "#FF9800"
    else:
        colour = NEG_COLOUR

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=ddm_value,
            delta={"reference": 0, "valueformat": ".3f"},
            gauge={
                "axis": {"range": [-1, 1], "tickcolor": "#ccc"},
                "bar": {"color": colour, "thickness": 0.25},
                "bgcolor": BG_COLOUR,
                "bordercolor": "#444",
                "steps": [
                    {"range": [-1, -0.55], "color": "#3d0000"},
                    {"range": [-0.55, -0.30], "color": "#3d1a00"},
                    {"range": [-0.30, 0.25], "color": "#1a1a2e"},
                    {"range": [0.25, 0.60], "color": "#002200"},
                    {"range": [0.60, 1], "color": "#003300"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 2},
                    "thickness": 0.8,
                    "value": ddm_value,
                },
            },
            title={"text": product, "font": {"color": "#ccc", "size": 13}},
        )
    )
    fig.update_layout(
        height=220,
        paper_bgcolor=BG_COLOUR,
        font={"color": "#ccc"},
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig
