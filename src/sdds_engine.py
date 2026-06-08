"""
sdds_engine.py
--------------
Computes the Sentiment Demand Direction Score (SDDS) per product entity
per time window using VADER sentiment analysis.

SDDS is a signed scalar in [-1, +1] where:
  +1.0  →  strongly positive consumer sentiment (demand growth predicted)
  -1.0  →  strongly negative consumer sentiment (demand decline predicted)
   0.0  →  neutral / insufficient signal volume
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Optional


class SDDSEngine:
    """
    Aspect-Based Sentiment Analysis engine for computing per-product
    Sentiment Demand Direction Scores from social media content.

    Parameters
    ----------
    volume_threshold : int
        Minimum number of posts required in a window to produce a valid
        SDDS. Below this threshold, SDDS defaults to 0.0 to suppress noise.
    window_minutes : int
        Rolling window size in minutes for SDDS aggregation.
    """

    def __init__(self, volume_threshold: int = 3, window_minutes: int = 60):
        self.analyzer = SentimentIntensityAnalyzer()
        self.volume_threshold = volume_threshold
        self.window_minutes = window_minutes

    def score_post(self, content: str) -> float:
        """
        Compute a raw sentiment score for a single post using VADER.

        VADER's compound score is already in [-1, +1]:
          >= +0.05  →  positive
          <= -0.05  →  negative
          between   →  neutral

        Returns
        -------
        float
            Signed sentiment score in [-1, +1].
        """
        scores = self.analyzer.polarity_scores(content)
        return round(scores["compound"], 4)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VADER-computed SDDS scores to a posts dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'content' column.

        Returns
        -------
        pd.DataFrame
            Original dataframe with added 'sdds' column.
        """
        df = df.copy()
        df["sdds"] = df["content"].apply(self.score_post)
        return df

    def compute_hourly_sdds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate post-level SDDS scores into hourly product-level scores.

        Applies volume threshold: if fewer than `volume_threshold` posts
        exist in a window, SDDS is set to 0.0.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'timestamp', 'product', 'sdds', 'likes', 'shares' columns.

        Returns
        -------
        pd.DataFrame
            Hourly SDDS per product with columns:
            [hour, product, sdds_mean, sdds_weighted, post_count,
             engagement_velocity, sdds_final]
        """
        df = df.copy()
        df["hour"] = df["timestamp"].dt.floor("h")

        # Engagement weight: normalise per-product so high-engagement posts
        # contribute more to the hourly SDDS
        df["engagement"] = df["likes"] + df["shares"] * 2
        df["engagement_norm"] = df.groupby(["product", "hour"])["engagement"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1 / len(x)
        )
        df["sdds_weighted_contrib"] = df["sdds"] * df["engagement_norm"]

        hourly = (
            df.groupby(["hour", "product"])
            .agg(
                sdds_mean=("sdds", "mean"),
                sdds_weighted=("sdds_weighted_contrib", "sum"),
                post_count=("sdds", "count"),
                engagement_velocity=("engagement", "sum"),
            )
            .reset_index()
        )

        # Apply volume threshold — suppress noise from low-volume windows
        hourly["sdds_final"] = np.where(
            hourly["post_count"] >= self.volume_threshold,
            hourly["sdds_weighted"].clip(-1, 1),
            0.0,
        )

        return hourly

    def get_sentiment_breakdown(self, df: pd.DataFrame, product: str) -> dict:
        """
        Return a breakdown of positive / neutral / negative post counts
        for a specific product.
        """
        subset = df[df["product"] == product]["sdds"]
        return {
            "positive": int((subset > 0.05).sum()),
            "neutral":  int(((subset >= -0.05) & (subset <= 0.05)).sum()),
            "negative": int((subset < -0.05).sum()),
            "total":    len(subset),
            "mean_sdds": round(subset.mean(), 3),
        }
