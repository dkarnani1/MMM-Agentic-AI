"""
sample_data.py — Synthetic MMM Dataset Generator

Generates a realistic 4-year weekly marketing dataset inspired by the
Google LightweightMMM sample dataset structure.

Dataset features:
  • 208 weeks (4 years) of weekly observations
  • 4 paid media channels: TV, Radio, Digital, Print
  • Realistic spend patterns with campaign bursts and seasonality
  • Sales driven by media effectiveness + seasonality + trend + noise
  • Known ground-truth model parameters for validation

Usage:
    from data.sample_data import generate_mmm_dataset
    df = generate_mmm_dataset()
"""

import numpy as np
import pandas as pd
from typing import Optional


def generate_mmm_dataset(
    n_weeks: int = 208,
    seed: int = 42,
    noise_scale: float = 150.0,
) -> pd.DataFrame:
    """
    Generate a synthetic Marketing Mix Model dataset.

    Args:
        n_weeks:     Number of weeks to generate (default 208 = 4 years).
        seed:        Random seed for reproducibility.
        noise_scale: Standard deviation of sales noise ($000s).

    Returns:
        pd.DataFrame with columns:
            date, week, year, quarter,
            sales,
            tv_spend, radio_spend, digital_spend, print_spend,
            trend, seasonality
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)  # Week index 0..207

    # ── Dates ──────────────────────────────────────────────────────────────
    dates = pd.date_range(start="2020-01-06", periods=n_weeks, freq="W-MON")

    # ── Media Spend ($000s) ─────────────────────────────────────────────────
    # TV: highest spend, strong seasonal swings, 3 major campaign bursts
    tv_base = 500 + 200 * np.sin(2 * np.pi * t / 52)
    tv_spend = np.maximum(0, tv_base + rng.normal(0, 80, n_weeks))
    tv_spend[40:55]   *= 2.5   # Spring campaign
    tv_spend[90:100]  *= 1.8   # Summer push
    tv_spend[140:160] *= 2.2   # Q4 holiday blitz
    tv_spend[192:208] *= 2.0   # Year-end

    # Radio: lower spend, moderate variation
    radio_base = 100 + 50 * np.sin(2 * np.pi * t / 52 + 0.5)
    radio_spend = np.maximum(0, radio_base + rng.normal(0, 20, n_weeks))
    radio_spend[50:65]  *= 2.0
    radio_spend[145:158] *= 1.5

    # Digital: second-highest spend, responsive to events
    digital_base = 300 + 150 * np.sin(2 * np.pi * t / 52 - 0.5)
    digital_spend = np.maximum(0, digital_base + rng.normal(0, 60, n_weeks))
    digital_spend[20:30]  *= 1.8
    digital_spend[80:90]  *= 2.0
    digital_spend[145:155] *= 2.5   # Black Friday / Cyber Monday

    # Print: lowest spend, steady
    print_base = 80 + 30 * np.sin(2 * np.pi * t / 52)
    print_spend = np.maximum(0, print_base + rng.normal(0, 15, n_weeks))

    # ── Exogenous Components ────────────────────────────────────────────────
    # Trend: mild linear growth over 4 years
    trend = t / n_weeks  # 0 → 1

    # Seasonality: annual + semi-annual harmonics
    seasonality = (
        0.15 * np.sin(2 * np.pi * t / 52)
        + 0.05 * np.sin(4 * np.pi * t / 52)
        + 0.03 * np.cos(2 * np.pi * t / 52)
    )

    # ── True Data Generating Process ───────────────────────────────────────
    # (These parameters match what FeatureEngineeringAgent will apply,
    #  so the OLS model can recover them cleanly.)
    def _adstock(x: np.ndarray, decay: float) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        out[0] = x[0]
        for i in range(1, len(x)):
            out[i] = x[i] + decay * out[i - 1]
        return out

    def _hill(x: np.ndarray, alpha: float, K: float) -> np.ndarray:
        max_val = x.max()
        if max_val == 0:
            return np.zeros_like(x)
        xn = x / max_val
        return xn**alpha / (xn**alpha + K**alpha)

    tv_sat      = _hill(_adstock(tv_spend,      0.70), 2.0, 0.50)
    radio_sat   = _hill(_adstock(radio_spend,   0.40), 1.5, 0.40)
    digital_sat = _hill(_adstock(digital_spend, 0.20), 2.5, 0.30)
    print_sat   = _hill(_adstock(print_spend,   0.50), 1.8, 0.45)

    BASE_SALES  = 5_000.0   # Organic baseline ($000s / week)
    COEF_TV     = 1_800.0
    COEF_RADIO  =   600.0
    COEF_DIGITAL = 1_400.0
    COEF_PRINT  =   400.0

    sales = (
        BASE_SALES
        + COEF_TV      * tv_sat
        + COEF_RADIO   * radio_sat
        + COEF_DIGITAL * digital_sat
        + COEF_PRINT   * print_sat
        + BASE_SALES * 0.15 * trend        # Growth bonus
        + BASE_SALES * seasonality
        + rng.normal(0, noise_scale, n_weeks)
    )
    sales = np.maximum(sales, 0.0)

    # ── Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "date":          dates,
        "week":          t + 1,
        "sales":         np.round(sales, 2),
        "tv_spend":      np.round(tv_spend, 2),
        "radio_spend":   np.round(radio_spend, 2),
        "digital_spend": np.round(digital_spend, 2),
        "print_spend":   np.round(print_spend, 2),
        "trend":         np.round(trend, 6),
        "seasonality":   np.round(seasonality, 6),
    })
    df["year"]    = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"]   = df["date"].dt.month

    return df


if __name__ == "__main__":
    df = generate_mmm_dataset()
    print(df.head(10).to_string(index=False))
    print(f"\nShape : {df.shape}")
    print(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nSales summary ($000s):\n{df['sales'].describe().round(1)}")
