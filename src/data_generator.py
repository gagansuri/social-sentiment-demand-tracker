"""
data_generator.py
-----------------
Generates synthetic social media posts for retail products
with realistic sentiment trajectories (viral growth, complaint spiral,
neutral plateau). No API keys required.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import random

# ── Product scenarios ─────────────────────────────────────────────────────────
PRODUCTS = {
    "Stanley Tumbler": {
        "scenario": "viral_positive",
        "base_sentiment": 0.65,
        "trend_slope": 0.008,
        "volatility": 0.12,
        "peak_volume_multiplier": 4.0,
        "platforms": ["TikTok", "Instagram", "Pinterest"],
    },
    "Crocs Classic": {
        "scenario": "plateauing",
        "base_sentiment": 0.30,
        "trend_slope": -0.002,
        "volatility": 0.10,
        "peak_volume_multiplier": 1.5,
        "platforms": ["TikTok", "Instagram", "Reddit"],
    },
    "Ninja Blender": {
        "scenario": "complaint_spiral",
        "base_sentiment": 0.10,
        "trend_slope": -0.012,
        "volatility": 0.15,
        "peak_volume_multiplier": 2.5,
        "platforms": ["Reddit", "X / Twitter", "YouTube"],
    },
    "Oura Ring": {
        "scenario": "recall_event",
        "base_sentiment": 0.40,
        "trend_slope": -0.020,
        "volatility": 0.18,
        "peak_volume_multiplier": 3.0,
        "platforms": ["Reddit", "X / Twitter", "Instagram"],
    },
    "Athletic Greens": {
        "scenario": "stable_positive",
        "base_sentiment": 0.50,
        "trend_slope": 0.001,
        "volatility": 0.08,
        "peak_volume_multiplier": 1.2,
        "platforms": ["Instagram", "YouTube", "Reddit"],
    },
}

# ── Positive post templates ───────────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "Just got my {product} and I'm obsessed! 10/10 recommend 🔥",
    "The {product} hype is real. Best purchase I've made all year.",
    "Why is everyone talking about {product}? I finally tried it and WOW.",
    "My {product} just arrived and it's even better than the reviews said.",
    "{product} just dropped and it's already sold out everywhere 😭",
    "Can't believe how good {product} is. Game changer for my routine.",
    "Added {product} to my cart and honestly best decision ever.",
    "The {product} is worth every penny. Absolutely love it.",
    "Everyone needs a {product} in their life, no joke.",
    "Just convinced my entire family to get {product}. We're obsessed.",
    "The {product} trend on TikTok got me and I have zero regrets.",
    "Rating my {product}: 5/5 stars. Exceeded all expectations.",
]

# ── Negative post templates ───────────────────────────────────────────────────
NEGATIVE_TEMPLATES = [
    "My {product} broke after 2 weeks. Total waste of money.",
    "Is anyone else having problems with their {product}? Mine is defective.",
    "Do NOT buy {product}. Worst purchase I've made this year.",
    "Returning my {product} tomorrow. Completely disappointed.",
    "The {product} is WAY overhyped. Not worth the price at all.",
    "Has anyone seen the new report about {product}? Very concerning.",
    "Trying to find a good alternative to {product}. Any suggestions?",
    "The {product} recall news is scary. Glad I returned mine.",
    "Why does {product} have such bad reviews all of a sudden?",
    "I've had nothing but problems with {product}. Stay away.",
    "Warning: {product} may have safety issues. Do your research.",
    "Finally returning my {product}. The complaints online were right.",
]

# ── Neutral post templates ────────────────────────────────────────────────────
NEUTRAL_TEMPLATES = [
    "Has anyone tried {product}? Thinking about buying it.",
    "Looking for reviews on {product} before I buy.",
    "How does {product} compare to similar products?",
    "Just saw an ad for {product}. Anyone have experience with it?",
    "Is {product} still popular or is it over?",
    "What's the current verdict on {product}?",
]


def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _sentiment_for_scenario(
    scenario: str,
    base: float,
    slope: float,
    volatility: float,
    t: int,
    total_steps: int,
) -> float:
    """Compute ground-truth sentiment score for a given time step."""
    if scenario == "viral_positive":
        # S-curve growth peaking around 60% through the period
        progress = t / total_steps
        trend = base + slope * t + 0.3 * (1 / (1 + np.exp(-10 * (progress - 0.4))))
    elif scenario == "recall_event":
        # Sharp negative shock at t=40% then sustained decline
        shock = -0.6 if t > total_steps * 0.4 else 0
        trend = base + slope * t + shock
    elif scenario == "complaint_spiral":
        # Gradual decline accelerating over time
        trend = base + slope * t * (1 + t / total_steps)
    elif scenario == "plateauing":
        # Slow decline from moderate positive
        trend = base + slope * t
    else:  # stable_positive
        trend = base + slope * t

    noise = np.random.normal(0, volatility)
    return _clamp(trend + noise)


def _pick_template(sentiment_score: float) -> str:
    """Pick a post template based on sentiment score."""
    if sentiment_score > 0.2:
        return random.choice(POSITIVE_TEMPLATES)
    elif sentiment_score < -0.2:
        return random.choice(NEGATIVE_TEMPLATES)
    else:
        return random.choice(NEUTRAL_TEMPLATES)


def generate_synthetic_posts(
    days: int = 7,
    posts_per_hour_base: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic social media post dataset.

    Parameters
    ----------
    days : int
        Number of days to simulate.
    posts_per_hour_base : int
        Base number of posts per product per hour (scales with scenario).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, product, platform, content, sentiment_score,
                 likes, shares, comments, saves
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_time = datetime.now() - timedelta(days=days)
    total_hours = days * 24
    records = []

    for product, cfg in PRODUCTS.items():
        scenario = cfg["scenario"]
        for hour in range(total_hours):
            ts = start_time + timedelta(hours=hour)

            # Volume scales with scenario momentum
            progress = hour / total_hours
            if scenario == "viral_positive":
                volume_mult = 1 + (cfg["peak_volume_multiplier"] - 1) * progress
            elif scenario in ("recall_event", "complaint_spiral"):
                volume_mult = 1 + (cfg["peak_volume_multiplier"] - 1) * progress * 0.7
            else:
                volume_mult = cfg["peak_volume_multiplier"] * 0.5

            n_posts = max(1, int(np.random.poisson(posts_per_hour_base * volume_mult)))

            for _ in range(n_posts):
                sent = _sentiment_for_scenario(
                    scenario,
                    cfg["base_sentiment"],
                    cfg["trend_slope"],
                    cfg["volatility"],
                    hour,
                    total_hours,
                )
                template = _pick_template(sent)
                content = template.format(product=product)
                platform = random.choice(cfg["platforms"])

                # Engagement scales with absolute sentiment and volume
                base_eng = abs(sent) * 500
                likes    = int(np.random.lognormal(np.log(max(1, base_eng)), 0.8))
                shares   = int(likes * np.random.uniform(0.05, 0.25))
                comments = int(likes * np.random.uniform(0.03, 0.15))
                saves    = int(likes * np.random.uniform(0.02, 0.10))

                records.append({
                    "timestamp":       ts + timedelta(minutes=random.randint(0, 59)),
                    "product":         product,
                    "platform":        platform,
                    "content":         content,
                    "ground_truth_sentiment": round(sent, 4),
                    "likes":           likes,
                    "shares":          shares,
                    "comments":        comments,
                    "saves":           saves,
                })

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_synthetic_posts(days=7)
    print(f"Generated {len(df):,} posts across {df['product'].nunique()} products")
    print(df.groupby("product")["ground_truth_sentiment"].agg(["mean", "min", "max"]).round(3))
    df.to_csv("data/synthetic_posts.csv", index=False)
    print("Saved to data/synthetic_posts.csv")
