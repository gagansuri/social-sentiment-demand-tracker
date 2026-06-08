"""
ddm_engine.py
-------------
Computes the Demand Direction Modifier (DDM) and Sentiment Velocity
from a time series of hourly SDDS scores.

DDM is a rolling exponentially-weighted average of SDDS in [-1, +1]:
  - Positive DDM  →  accelerate replenishment
  - Negative DDM  →  suppress orders, flag markdown / return-to-supplier

Sentiment Velocity is the first temporal derivative of the rolling SDDS,
detecting ACCELERATING deterioration before the DDM crosses a threshold.
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ── Inventory action thresholds ───────────────────────────────────────────────
THRESHOLD_REPLENISH_STRONG  =  0.60   # DDM above: accelerate replenishment
THRESHOLD_REPLENISH_MILD    =  0.25   # DDM above: standard replenishment
THRESHOLD_SUPPRESS          = -0.30   # DDM below: suppress next order
THRESHOLD_MARKDOWN          = -0.55   # DDM below: recommend markdown
THRESHOLD_RETURN_SUPPLIER   = -0.70   # DDM below: return-to-supplier alert
THRESHOLD_VELOCITY_ALERT    = -0.08   # velocity below: early warning escalation


class DDMEngine:
    """
    Computes the Demand Direction Modifier (DDM) and Sentiment Velocity
    for each product from hourly SDDS time series.

    Parameters
    ----------
    decay_rate : float
        Exponential decay rate λ for the rolling weighted average.
        Higher values weight recent data more heavily. Default 0.15.
    lookback_hours : int
        Number of hours to include in the rolling DDM window. Default 24.
    velocity_window : int
        Number of hours for sentiment velocity computation. Default 4.
    """

    def __init__(
        self,
        decay_rate: float = 0.15,
        lookback_hours: int = 24,
        velocity_window: int = 4,
    ):
        self.decay_rate = decay_rate
        self.lookback_hours = lookback_hours
        self.velocity_window = velocity_window

    def _exponential_weights(self, n: int) -> np.ndarray:
        """Compute exponential decay weights for n time steps, most recent last."""
        ages = np.arange(n - 1, -1, -1)  # oldest = n-1, newest = 0
        weights = np.exp(-self.decay_rate * ages)
        return weights / weights.sum()

    def compute_ddm_series(self, hourly_sdds: pd.DataFrame) -> pd.DataFrame:
        """
        Compute DDM and Sentiment Velocity for each product over time.

        Parameters
        ----------
        hourly_sdds : pd.DataFrame
            Output of SDDSEngine.compute_hourly_sdds().
            Must have columns: [hour, product, sdds_final]

        Returns
        -------
        pd.DataFrame
            Columns: [hour, product, sdds_final, ddm, sentiment_velocity,
                      action, action_color, action_emoji]
        """
        results = []

        for product in hourly_sdds["product"].unique():
            prod_df = (
                hourly_sdds[hourly_sdds["product"] == product]
                .sort_values("hour")
                .reset_index(drop=True)
            )

            ddm_values = []
            velocity_values = []

            for i in range(len(prod_df)):
                # DDM: exponentially weighted rolling average
                window_start = max(0, i - self.lookback_hours + 1)
                window = prod_df["sdds_final"].iloc[window_start : i + 1].values
                weights = self._exponential_weights(len(window))
                ddm = float(np.clip(np.dot(weights, window), -1, 1))
                ddm_values.append(round(ddm, 4))

                # Sentiment velocity: slope of SDDS over velocity_window
                vel_start = max(0, i - self.velocity_window + 1)
                vel_window = prod_df["sdds_final"].iloc[vel_start : i + 1].values
                if len(vel_window) >= 2:
                    x = np.arange(len(vel_window))
                    slope = float(np.polyfit(x, vel_window, 1)[0])
                else:
                    slope = 0.0
                velocity_values.append(round(slope, 5))

            prod_df["ddm"] = ddm_values
            prod_df["sentiment_velocity"] = velocity_values
            results.append(prod_df)

        result_df = pd.concat(results, ignore_index=True)
        result_df = self._assign_actions(result_df)
        return result_df

    def _assign_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign inventory action recommendations based on DDM and velocity."""
        actions, colors, emojis = [], [], []

        for _, row in df.iterrows():
            ddm = row["ddm"]
            vel = row["sentiment_velocity"]

            # Velocity early-warning escalation
            if vel < THRESHOLD_VELOCITY_ALERT and ddm < 0:
                action = "⚡ EARLY WARNING — Accelerating decline"
                color  = "#ff4444"
                emoji  = "⚡"
            elif ddm <= THRESHOLD_RETURN_SUPPLIER:
                action = "🔴 RETURN TO SUPPLIER — Critical sentiment"
                color  = "#cc0000"
                emoji  = "🔴"
            elif ddm <= THRESHOLD_MARKDOWN:
                action = "🟠 RECOMMEND MARKDOWN — Negative sentiment"
                color  = "#ff6600"
                emoji  = "🟠"
            elif ddm <= THRESHOLD_SUPPRESS:
                action = "🟡 SUPPRESS NEXT ORDER — Declining sentiment"
                color  = "#ffaa00"
                emoji  = "🟡"
            elif ddm >= THRESHOLD_REPLENISH_STRONG:
                action = "🟢 ACCELERATE REPLENISHMENT — Strong positive"
                color  = "#00aa44"
                emoji  = "🟢"
            elif ddm >= THRESHOLD_REPLENISH_MILD:
                action = "🟢 STANDARD REPLENISHMENT — Positive sentiment"
                color  = "#44cc66"
                emoji  = "🟢"
            else:
                action = "⚪ MONITOR — Neutral signal"
                color  = "#888888"
                emoji  = "⚪"

            actions.append(action)
            colors.append(color)
            emojis.append(emoji)

        df["action"]       = actions
        df["action_color"] = colors
        df["action_emoji"] = emojis
        return df

    def get_latest_signals(self, ddm_series: pd.DataFrame) -> pd.DataFrame:
        """
        Return the most recent DDM signal per product — the operational
        output that would be fed into the inventory system.
        """
        latest = (
            ddm_series.sort_values("hour")
            .groupby("product")
            .last()
            .reset_index()
        )[["product", "sdds_final", "ddm", "sentiment_velocity", "action", "action_emoji"]]
        latest.columns = [
            "Product", "SDDS (latest)", "DDM", "Sentiment Velocity",
            "Recommended Action", "Status"
        ]
        latest["SDDS (latest)"] = latest["SDDS (latest)"].round(3)
        latest["DDM"]           = latest["DDM"].round(3)
        latest["Sentiment Velocity"] = latest["Sentiment Velocity"].round(4)
        return latest.sort_values("DDM", ascending=False).reset_index(drop=True)
