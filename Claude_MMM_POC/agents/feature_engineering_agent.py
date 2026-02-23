"""
feature_engineering_agent.py — Agent 3: Feature Engineering

Responsibilities:
  * Apply adstock (geometric decay) to each media channel
  * Apply Hill-function saturation to capture diminishing returns
  * Add trend and Fourier-based seasonality components
  * Return an enriched DataFrame ready for regression

Key Transformations:
  1. Adstock   : adstock[t] = spend[t] + λ × adstock[t-1]
  2. Saturation: f(x) = x^α / (x^α + K^α)   [Hill function]
  3. Trend     : linear 0→1 across the dataset
  4. Seasonality: sin/cos Fourier terms at annual frequency
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from config.settings import (
    MEDIA_CHANNELS,
    SPEND_COLS,
    ADSTOCK_DECAY_RATES,
    SATURATION_PARAMS,
)


class FeatureEngineeringAgent:
    """
    Transforms raw spend data into model-ready features.

    Usage:
        agent = FeatureEngineeringAgent()
        df_engineered = agent.engineer_all_features(df_raw)
    """

    def __init__(
        self,
        decay_rates: Optional[Dict[str, float]] = None,
        sat_params:  Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Args:
            decay_rates: Override adstock decay rates (channel → rate).
            sat_params:  Override saturation params (channel → {alpha, K}).
        """
        self.decay_rates = decay_rates or ADSTOCK_DECAY_RATES
        self.sat_params  = sat_params  or SATURATION_PARAMS
        self._adstock_max: Dict[str, float] = {}   # stored for response curves

    # == Public Interface ====================================================

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformations and return enriched DataFrame.

        New columns added:
            {ch}_adstock       — adstocked spend
            {ch}_transformed   — adstock + saturation (model input)
            trend              — linear 0→1
            sin_annual, cos_annual — annual seasonality harmonics
            sin_semi, cos_semi     — semi-annual harmonics

        Args:
            df: Raw dataset from DataIngestionAgent.

        Returns:
            Enriched DataFrame (original columns preserved).
        """
        df = df.copy()
        n  = len(df)

        for ch in MEDIA_CHANNELS:
            spend_col = SPEND_COLS[ch]
            if spend_col not in df.columns:
                raise KeyError(
                    f"Expected column '{spend_col}' not found in DataFrame."
                )
            spend = df[spend_col].to_numpy(dtype=float)
            adstocked = self._apply_adstock(spend, self.decay_rates[ch])
            saturated = self._apply_saturation(
                adstocked,
                self.sat_params[ch]["alpha"],
                self.sat_params[ch]["K"],
            )
            self._adstock_max[ch] = adstocked.max()
            df[f"{ch.lower()}_adstock"]     = np.round(adstocked, 4)
            df[f"{ch.lower()}_transformed"] = np.round(saturated, 6)

        # Trend: linear 0 → 1
        df["trend"] = np.linspace(0, 1, n)

        # Fourier seasonality (week-of-year proxy using index)
        week_idx = np.arange(n, dtype=float)
        df["sin_annual"]  = np.sin(2 * np.pi * week_idx / 52)
        df["cos_annual"]  = np.cos(2 * np.pi * week_idx / 52)
        df["sin_semi"]    = np.sin(4 * np.pi * week_idx / 52)
        df["cos_semi"]    = np.cos(4 * np.pi * week_idx / 52)

        return df

    def get_transformed_col_names(self) -> list[str]:
        """Return the list of transformed feature column names."""
        cols = [f"{ch.lower()}_transformed" for ch in MEDIA_CHANNELS]
        cols += ["trend", "sin_annual", "cos_annual", "sin_semi", "cos_semi"]
        return cols

    # == Core Math ===========================================================

    @staticmethod
    def _apply_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
        """
        Geometric (Koyck) adstock transformation.

        Captures advertising carryover: past spend continues to
        influence future sales at a geometrically decaying rate.

        Args:
            spend: Raw weekly spend array.
            decay: Decay rate λ ∈ [0, 1).  Higher = longer memory.

        Returns:
            Adstocked spend array (same length as input).
        """
        result = np.empty_like(spend, dtype=float)
        result[0] = spend[0]
        for t in range(1, len(spend)):
            result[t] = spend[t] + decay * result[t - 1]
        return result

    @staticmethod
    def _apply_saturation(
        adstocked: np.ndarray,
        alpha: float,
        K: float,
    ) -> np.ndarray:
        """
        Hill-function saturation transformation.

        Captures diminishing marginal returns:
            f(x) = x^α / (x^α + K^α),  x ∈ [0, 1]  (normalised)

        At K: exactly 50% of maximum response.
        alpha > 1 produces an S-curve; alpha ≤ 1 is concave (pure decay).

        Args:
            adstocked: Adstock-transformed spend array.
            alpha:     Steepness parameter.
            K:         Half-saturation point (fraction of max).

        Returns:
            Saturation-transformed array with values in [0, 1].
        """
        max_val = adstocked.max()
        if max_val == 0:
            return np.zeros_like(adstocked, dtype=float)
        x = adstocked / max_val
        return x**alpha / (x**alpha + K**alpha)
