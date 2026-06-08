"""
demand_signal.py
----------------
Main pipeline class that orchestrates the full signal processing flow:
  1. Generate / load synthetic social media posts
  2. Score each post with SDDS (VADER ABSA)
  3. Aggregate to hourly product-level SDDS
  4. Compute DDM and Sentiment Velocity
  5. Return operational inventory action signals
"""

import pandas as pd
from src.data_generator import generate_synthetic_posts
from src.sdds_engine import SDDSEngine
from src.ddm_engine import DDMEngine


class DemandSignalPipeline:
    """
    End-to-end social sentiment demand signal pipeline.

    Parameters
    ----------
    days : int
        Number of days of synthetic data to generate.
    volume_threshold : int
        Minimum posts per hour to produce a valid SDDS (default 3).
    decay_rate : float
        DDM exponential decay rate (default 0.15).
    lookback_hours : int
        DDM rolling window in hours (default 24).
    """

    def __init__(
        self,
        days: int = 7,
        volume_threshold: int = 3,
        decay_rate: float = 0.15,
        lookback_hours: int = 24,
        random_seed: int = 42,
    ):
        self.days = days
        self.sdds_engine = SDDSEngine(volume_threshold=volume_threshold)
        self.ddm_engine  = DDMEngine(decay_rate=decay_rate, lookback_hours=lookback_hours)
        self.random_seed = random_seed

        # Pipeline state
        self.raw_posts:    pd.DataFrame | None = None
        self.scored_posts: pd.DataFrame | None = None
        self.hourly_sdds:  pd.DataFrame | None = None
        self.ddm_series:   pd.DataFrame | None = None
        self.latest:       pd.DataFrame | None = None

    def run(self) -> "DemandSignalPipeline":
        """Execute the full pipeline and return self for chaining."""
        print("📥  Generating synthetic social media data...")
        self.raw_posts = generate_synthetic_posts(
            days=self.days, random_seed=self.random_seed
        )
        print(f"    {len(self.raw_posts):,} posts across {self.raw_posts['product'].nunique()} products\n")

        print("🧠  Scoring posts with SDDS (ABSA)...")
        self.scored_posts = self.sdds_engine.score_dataframe(self.raw_posts)
        print(f"    Sentiment range: [{self.scored_posts['sdds'].min():.3f}, "
              f"{self.scored_posts['sdds'].max():.3f}]\n")

        print("📊  Aggregating to hourly SDDS per product...")
        self.hourly_sdds = self.sdds_engine.compute_hourly_sdds(self.scored_posts)
        print(f"    {len(self.hourly_sdds):,} hourly observations\n")

        print("📈  Computing DDM and Sentiment Velocity...")
        self.ddm_series = self.ddm_engine.compute_ddm_series(self.hourly_sdds)

        print("✅  Pipeline complete. Latest signals:\n")
        self.latest = self.ddm_engine.get_latest_signals(self.ddm_series)
        print(self.latest.to_string(index=False))
        return self

    def get_product_series(self, product: str) -> pd.DataFrame:
        """Return the full time series for a specific product."""
        if self.ddm_series is None:
            raise RuntimeError("Run pipeline first with .run()")
        return self.ddm_series[self.ddm_series["product"] == product].copy()

    def get_sentiment_breakdown(self, product: str) -> dict:
        """Return positive/neutral/negative post breakdown for a product."""
        if self.scored_posts is None:
            raise RuntimeError("Run pipeline first with .run()")
        return self.sdds_engine.get_sentiment_breakdown(self.scored_posts, product)
