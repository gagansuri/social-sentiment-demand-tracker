"""
run_demo.py
-----------
CLI demo of the Social Sentiment Demand Tracker pipeline.
Runs the full pipeline, prints results to terminal, and saves
outputs to data/ directory.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --days 14 --seed 99
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.demand_signal import DemandSignalPipeline
from src.visualizer import plot_sdds_timeseries, plot_ddm_comparison

os.makedirs("data", exist_ok=True)


def main(days: int = 7, seed: int = 42):
    print("=" * 65)
    print("   SOCIAL SENTIMENT DEMAND TRACKER")
    print("   Patent Pending — DSFE-2026-001")
    print("=" * 65)

    # ── Run pipeline ──────────────────────────────────────────────
    pipe = DemandSignalPipeline(days=days, random_seed=seed).run()

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("OPERATIONAL SIGNALS — Latest DDM per Product")
    print("=" * 65)
    print(pipe.latest.to_string(index=False))

    # ── Per-product breakdown ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("SENTIMENT BREAKDOWN PER PRODUCT")
    print("=" * 65)
    for product in pipe.scored_posts["product"].unique():
        b = pipe.get_sentiment_breakdown(product)
        pct_neg = 100 * b["negative"] / b["total"] if b["total"] else 0
        pct_pos = 100 * b["positive"] / b["total"] if b["total"] else 0
        print(
            f"\n  {product:<22}"
            f"  Posts: {b['total']:>5,}"
            f"  +{pct_pos:4.1f}%  -{pct_neg:4.1f}%"
            f"  Mean SDDS: {b['mean_sdds']:+.3f}"
        )

    # ── Save outputs ───────────────────────────────────────────────
    pipe.scored_posts.to_csv("data/scored_posts.csv", index=False)
    pipe.hourly_sdds.to_csv("data/hourly_sdds.csv", index=False)
    pipe.ddm_series.to_csv("data/ddm_series.csv", index=False)
    pipe.latest.to_csv("data/latest_signals.csv", index=False)

    print("\n✅  Outputs saved to data/")
    print("    scored_posts.csv   — all posts with SDDS scores")
    print("    hourly_sdds.csv    — hourly aggregated SDDS")
    print("    ddm_series.csv     — DDM + velocity time series")
    print("    latest_signals.csv — current action signals")

    # ── Save HTML charts ───────────────────────────────────────────
    os.makedirs("data/charts", exist_ok=True)
    fig_comparison = plot_ddm_comparison(pipe.latest)
    fig_comparison.write_html("data/charts/ddm_comparison.html")

    for product in pipe.scored_posts["product"].unique():
        safe = product.replace(" ", "_").lower()
        fig = plot_sdds_timeseries(pipe.ddm_series, product)
        fig.write_html(f"data/charts/sdds_{safe}.html")

    print("    charts/            — interactive HTML charts")
    print("\n🚀  Run the Streamlit app:")
    print("    streamlit run app/streamlit_app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Social Sentiment Demand Tracker demo")
    parser.add_argument("--days", type=int, default=7, help="Days to simulate (default 7)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args()
    main(days=args.days, seed=args.seed)
