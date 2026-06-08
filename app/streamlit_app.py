"""
streamlit_app.py
----------------
Interactive Streamlit demo for the Social Sentiment Demand Tracker.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.demand_signal import DemandSignalPipeline
from src.visualizer import (
    plot_sdds_timeseries,
    plot_ddm_comparison,
    plot_sentiment_heatmap,
    plot_post_volume,
    plot_gauge,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Sentiment Demand Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e; border-radius: 10px;
        padding: 16px; text-align: center; border: 1px solid #2a2a4e;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.8rem; color: #aaa; margin-top: 4px; }
    .action-badge {
        padding: 6px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/Patent-Pending-green.svg")
    st.title("⚙️ Configuration")

    days = st.slider("Simulation days", 3, 14, 7)
    decay_rate = st.slider("DDM decay rate (λ)", 0.05, 0.50, 0.15, 0.01)
    lookback = st.slider("DDM lookback (hours)", 6, 48, 24)
    volume_thresh = st.slider("Volume threshold (posts/hr)", 1, 10, 3)
    seed = st.number_input("Random seed", value=42, step=1)

    st.divider()
    st.markdown("### 📖 How it works")
    st.markdown("""
**SDDS** *(Sentiment Demand Direction Score)*
A signed score [-1, +1] computed per post using VADER sentiment analysis.

**DDM** *(Demand Direction Modifier)*
Exponentially-weighted rolling average of SDDS. Feeds the inventory action engine.

**Sentiment Velocity**
First derivative of SDDS — detects *accelerating* decline before DDM crosses threshold.
""")
    st.divider()
    st.markdown("*Part of the [AI in Retail](https://linkedin.com/in/gagansuri) patent-pending system*")
    st.markdown("**Patent Pending** — USPTO Provisional filed June 5, 2026")

# ── Run pipeline ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(days, decay_rate, lookback, volume_thresh, seed):
    pipe = DemandSignalPipeline(
        days=days,
        volume_threshold=volume_thresh,
        decay_rate=decay_rate,
        lookback_hours=lookback,
        random_seed=int(seed),
    )
    pipe.run()
    return pipe

with st.spinner("🔄 Running signal pipeline..."):
    pipe = run_pipeline(days, decay_rate, lookback, volume_thresh, seed)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Social Sentiment Demand Tracker")
st.markdown(
    "Real-time demand direction signals from social media — "
    "**before trends materialise in sales data.** *(Patent Pending)*"
)
st.divider()

# ── Latest signals table ─────────────────────────────────────────────────────
st.subheader("🚦 Current Inventory Action Signals")
latest = pipe.latest.copy()

# Colour-code the table
def colour_ddm(val):
    if val >= 0.25:  return "color: #00C853; font-weight: bold"
    if val <= -0.30: return "color: #D50000; font-weight: bold"
    return "color: #9E9E9E"

styled = latest.style.applymap(colour_ddm, subset=["DDM"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── Gauge row ─────────────────────────────────────────────────────────────────
st.subheader("📡 DDM Gauges")
gauge_cols = st.columns(len(latest))
for col, (_, row) in zip(gauge_cols, latest.iterrows()):
    with col:
        fig = plot_gauge(row["DDM"], row["Product"])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ── Overview charts ───────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📊 DDM Comparison")
    st.plotly_chart(
        plot_ddm_comparison(latest),
        use_container_width=True,
        config={"displayModeBar": False},
    )
with col2:
    st.subheader("🌡️ Sentiment Heatmap")
    st.plotly_chart(
        plot_sentiment_heatmap(pipe.hourly_sdds),
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.divider()

# ── Product deep-dive ─────────────────────────────────────────────────────────
st.subheader("🔍 Product Deep-Dive")
products = pipe.scored_posts["product"].unique().tolist()
selected = st.selectbox("Select product", products)

series = pipe.get_product_series(selected)
latest_row = series.sort_values("hour").iloc[-1]

# Metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest SDDS",    f"{latest_row['sdds_final']:+.3f}")
m2.metric("DDM",            f"{latest_row['ddm']:+.3f}")
m3.metric("Velocity",       f"{latest_row['sentiment_velocity']:+.4f}")
m4.metric("Action",         latest_row["action_emoji"])

# Time series chart
st.plotly_chart(
    plot_sdds_timeseries(pipe.ddm_series, selected),
    use_container_width=True,
)

# Sentiment breakdown
st.subheader("📝 Post Sentiment Breakdown")
breakdown = pipe.get_sentiment_breakdown(selected)
b1, b2, b3, b4 = st.columns(4)
b1.metric("Total Posts", f"{breakdown['total']:,}")
b2.metric("✅ Positive",  f"{breakdown['positive']:,}")
b3.metric("⚪ Neutral",   f"{breakdown['neutral']:,}")
b4.metric("❌ Negative",  f"{breakdown['negative']:,}")

st.divider()

# ── Platform volume ───────────────────────────────────────────────────────────
st.subheader("📱 Post Volume by Platform")
st.plotly_chart(
    plot_post_volume(pipe.scored_posts),
    use_container_width=True,
)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("🗄️ View raw scored posts (sample)"):
    sample = pipe.scored_posts.sample(min(200, len(pipe.scored_posts)), random_state=1)
    st.dataframe(
        sample[["timestamp","product","platform","content","sdds","likes","shares"]],
        use_container_width=True,
    )

st.divider()
st.caption(
    "Social Sentiment Demand Tracker · Patent Pending (USPTO Provisional, June 5 2026) · "
    "Built by [Gagan Suri](https://linkedin.com/in/gagansuri)"
)
