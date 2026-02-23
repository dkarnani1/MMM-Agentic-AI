"""
modeling_agent.py — Agent 4: Marketing Mix Model (Regression)

Responsibilities:
  * Fit an OLS regression on transformed media + control features
  * Calculate per-channel sales contributions (weekly and total)
  * Calculate ROI: incremental sales / media spend
  * Generate response curves (saturation / S-curves)
  * Simulate budget scenarios (what-if analysis)

Model:
    sales = β₀
          + Σ βᵢ × channel_i_transformed
          + β_trend × trend
          + β_sin1 × sin_annual + β_cos1 × cos_annual
          + β_sin2 × sin_semi   + β_cos2 × cos_semi
          + ε

Note: Uses statsmodels OLS for full coefficient statistics
      (p-values, confidence intervals, R²).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Optional, Tuple

from config.settings import (
    MEDIA_CHANNELS,
    SPEND_COLS,
    ADSTOCK_DECAY_RATES,
    SATURATION_PARAMS,
    RESPONSE_CURVE_POINTS,
)


class ModelingAgent:
    """
    Fits and interrogates the Marketing Mix Model.

    After calling fit(), the following attributes are populated:
        model_result     — statsmodels RegressionResultsWrapper
        contributions    — Dict[channel, pd.Series]  weekly $ contributions
        roi              — Dict[channel, float]       $ per $ spent
        decomposition    — pd.DataFrame               full decomposition table
    """

    FEATURE_COLS = (
        [f"{ch.lower()}_transformed" for ch in MEDIA_CHANNELS]
        + ["trend", "sin_annual", "cos_annual", "sin_semi", "cos_semi"]
    )

    def __init__(self) -> None:
        self.model_result  = None
        self.df: Optional[pd.DataFrame] = None
        self.contributions: Dict[str, pd.Series] = {}
        self.roi:           Dict[str, float]      = {}
        self.decomposition: Optional[pd.DataFrame] = None

    # == Public Interface ====================================================

    def fit(self, df: pd.DataFrame) -> "ModelingAgent":
        """
        Fit the OLS Marketing Mix Model.

        Args:
            df: Feature-engineered DataFrame from FeatureEngineeringAgent.

        Returns:
            self (for method chaining).
        """
        self.df = df.copy()

        # Validate features exist
        missing = [c for c in self.FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. "
                "Run FeatureEngineeringAgent first."
            )

        X = df[self.FEATURE_COLS]
        y = df["sales"]
        X_const = sm.add_constant(X, has_constant="add")

        self.model_result = sm.OLS(y, X_const).fit()

        self._calculate_contributions()
        self._calculate_roi()
        self._build_decomposition()

        return self

    def get_model_summary(self) -> str:
        """Return statsmodels OLS summary as a string."""
        if self.model_result is None:
            return "Model not fitted yet."
        return str(self.model_result.summary())

    def get_fit_quality(self) -> Dict[str, float]:
        """Return key goodness-of-fit metrics."""
        if self.model_result is None:
            return {}
        return {
            "R²":            round(self.model_result.rsquared, 4),
            "Adj. R²":       round(self.model_result.rsquared_adj, 4),
            "MAPE (%)":      round(self._mape(), 2),
            "N observations": int(self.model_result.nobs),
        }

    def get_response_curves(self) -> Dict[str, Dict]:
        """
        Generate saturation / response curves for each media channel.

        Returns:
            Dict keyed by channel with:
                spend          — spend range array ($000s)
                sales_lift     — incremental sales ($000s) from that spend level
                current_spend  — channel's actual mean weekly spend
                current_lift   — model contribution at current mean spend
        """
        if self.df is None or self.model_result is None:
            return {}

        curves = {}
        for ch in MEDIA_CHANNELS:
            spend_col = SPEND_COLS[ch]
            trans_col = f"{ch.lower()}_transformed"
            coef      = self.model_result.params.get(trans_col, 0.0)

            max_spend  = self.df[spend_col].max() * 1.5
            spend_range = np.linspace(0, max_spend, RESPONSE_CURVE_POINTS)

            # Steady-state adstock approximation: x / (1 - λ)
            decay   = ADSTOCK_DECAY_RATES[ch]
            ads_ss  = spend_range / max(1e-9, 1 - decay)
            ads_max = self.df[spend_col].max() / max(1e-9, 1 - decay)

            # Hill saturation on normalised adstock
            alpha = SATURATION_PARAMS[ch]["alpha"]
            K     = SATURATION_PARAMS[ch]["K"]
            x_norm = np.clip(ads_ss / max(ads_max, 1e-9), 0, 10)
            sat    = x_norm**alpha / (x_norm**alpha + K**alpha)

            sales_lift = coef * sat

            current_spend = self.df[spend_col].mean()
            current_ads   = current_spend / max(1e-9, 1 - decay)
            current_norm  = np.clip(current_ads / max(ads_max, 1e-9), 0, 10)
            current_sat   = current_norm**alpha / (current_norm**alpha + K**alpha)
            current_lift  = coef * current_sat

            curves[ch] = {
                "spend":         spend_range,
                "sales_lift":    sales_lift,
                "current_spend": current_spend,
                "current_lift":  current_lift,
            }

        return curves

    def simulate_budget_change(
        self,
        channel: str,
        pct_change: float,
    ) -> Dict[str, float]:
        """
        Estimate the impact of changing a channel's weekly budget.

        Uses the fitted model's saturation curve to account for
        diminishing / increasing returns at the new spend level.

        Args:
            channel:    Channel name (e.g., "TV").
            pct_change: Fractional change, e.g. -0.20 for a 20% cut.

        Returns:
            Scenario dict with old/new spend, delta sales, delta ROI.
        """
        if self.df is None or self.model_result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if channel not in MEDIA_CHANNELS:
            raise ValueError(f"Unknown channel: {channel}")

        trans_col = f"{channel.lower()}_transformed"
        spend_col = SPEND_COLS[channel]
        coef      = self.model_result.params.get(trans_col, 0.0)
        decay     = ADSTOCK_DECAY_RATES[channel]
        alpha     = SATURATION_PARAMS[channel]["alpha"]
        K         = SATURATION_PARAMS[channel]["K"]
        ads_max   = self.df[spend_col].max() / max(1e-9, 1 - decay)

        def _contribution(spend_level: float) -> float:
            ads_ss  = spend_level / max(1e-9, 1 - decay)
            x_norm  = np.clip(ads_ss / max(ads_max, 1e-9), 0, 10)
            sat     = x_norm**alpha / (x_norm**alpha + K**alpha)
            return coef * sat

        current_spend = self.df[spend_col].mean()
        new_spend     = max(0, current_spend * (1 + pct_change))

        current_weekly_contrib = _contribution(current_spend)
        new_weekly_contrib     = _contribution(new_spend)
        n_weeks                = len(self.df)

        delta_annual_sales   = (new_weekly_contrib - current_weekly_contrib) * n_weeks
        current_annual_spend = current_spend * n_weeks
        new_annual_spend     = new_spend * n_weeks
        delta_spend          = new_annual_spend - current_annual_spend

        new_roi = (
            new_weekly_contrib / new_spend if new_spend > 0 else 0.0
        )

        return {
            "channel":               channel,
            "pct_change":            pct_change * 100,
            "current_weekly_spend":  round(current_spend, 1),
            "new_weekly_spend":      round(new_spend, 1),
            "delta_annual_spend":    round(delta_spend, 1),
            "delta_annual_sales":    round(delta_annual_sales, 1),
            "sales_per_dollar_spent": round(
                delta_annual_sales / abs(delta_spend) if delta_spend != 0 else 0, 2
            ),
            "new_channel_roi":       round(new_roi, 4),
        }

    # == Private Helpers =====================================================

    def _calculate_contributions(self) -> None:
        """Decompose fitted sales into per-channel weekly contributions."""
        params = self.model_result.params
        df     = self.df

        for ch in MEDIA_CHANNELS:
            col  = f"{ch.lower()}_transformed"
            coef = params.get(col, 0.0)
            self.contributions[ch] = coef * df[col]

        # Control / base contributions
        base = pd.Series(params.get("const", 0.0), index=df.index)
        for ctrl in ["trend", "sin_annual", "cos_annual", "sin_semi", "cos_semi"]:
            if ctrl in params and ctrl in df.columns:
                base = base + params[ctrl] * df[ctrl]
        self.contributions["Base"] = base

    def _calculate_roi(self) -> None:
        """ROI = total sales contribution / total spend per channel."""
        for ch in MEDIA_CHANNELS:
            spend_col = SPEND_COLS[ch]
            total_contribution = self.contributions[ch].sum()
            total_spend        = self.df[spend_col].sum()
            self.roi[ch] = (
                total_contribution / total_spend if total_spend > 0 else 0.0
            )

    def _build_decomposition(self) -> None:
        """Build a full week-by-week sales decomposition table."""
        records = {"date": self.df["date"]}
        for label, series in self.contributions.items():
            records[label] = series.round(2)
        records["fitted"]  = self.model_result.fittedvalues.round(2)
        records["actual"]  = self.df["sales"].round(2)
        records["residual"] = self.model_result.resid.round(2)
        self.decomposition = pd.DataFrame(records)

    def _mape(self) -> float:
        """Mean Absolute Percentage Error (%)."""
        actual  = self.df["sales"]
        fitted  = self.model_result.fittedvalues
        mask    = actual != 0
        return float(np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100)
