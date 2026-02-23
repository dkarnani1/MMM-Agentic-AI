"""
connectors/base.py — Abstract Base Connector

All live data connectors inherit from BaseConnector and must implement
fetch_to_dataframe(), returning a weekly-grain DataFrame that matches
the schema expected by DataIngestionAgent.

Schema (REQUIRED_COLUMNS):
    date            — week start date (datetime)
    sales           — weekly revenue ($000s)
    tv_spend        — weekly TV media spend ($000s)
    radio_spend     — weekly Radio media spend ($000s)
    digital_spend   — weekly Digital media spend ($000s)
    print_spend     — weekly Print media spend ($000s)

Notes:
  * Connectors are responsible for aggregating daily API data to weekly grain.
  * Missing channels should be filled with 0.0.
  * The fallback_to_synthetic() method returns a minimal valid DataFrame for
    local development when credentials are not configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "date", "sales", "tv_spend", "radio_spend", "digital_spend", "print_spend"
]


class BaseConnector(ABC):
    """
    Abstract base class for all MMM data connectors.

    Subclasses must implement:
        fetch_to_dataframe(start, end) -> pd.DataFrame
    """

    # == Public Interface ====================================================

    @abstractmethod
    def fetch_to_dataframe(self, start: date, end: date) -> pd.DataFrame:
        """
        Fetch data from the external API and return a weekly MMM DataFrame.

        Args:
            start: First date of the fetch window (inclusive).
            end:   Last date of the fetch window (inclusive).

        Returns:
            Weekly-grain DataFrame with REQUIRED_COLUMNS.
            Rows must be sorted by date ascending.
        """
        ...

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Check that a DataFrame has all required MMM columns.

        Args:
            df: DataFrame to validate.

        Returns:
            True if all REQUIRED_COLUMNS are present, False otherwise.
        """
        return all(c in df.columns for c in REQUIRED_COLUMNS)

    def is_configured(self) -> bool:
        """
        Return True if the connector's credentials are available.

        Default implementation always returns False.
        Subclasses should override to check their specific env vars.
        """
        return False

    # == Protected Helpers ===================================================

    def _date_range_weeks(self, start: date, end: date) -> pd.DatetimeIndex:
        """Return weekly Monday-anchored DatetimeIndex covering [start, end]."""
        return pd.date_range(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            freq="W-MON",
        )

    def _fallback_synthetic(self, start: date, end: date) -> pd.DataFrame:
        """
        Return a zero-spend, zero-sales DataFrame covering the date range.

        Used when credentials are absent so the rest of the pipeline
        can still be exercised without live API access.
        """
        weeks = self._date_range_weeks(start, end)
        return pd.DataFrame(
            {
                "date":          weeks,
                "sales":         np.zeros(len(weeks)),
                "tv_spend":      np.zeros(len(weeks)),
                "radio_spend":   np.zeros(len(weeks)),
                "digital_spend": np.zeros(len(weeks)),
                "print_spend":   np.zeros(len(weeks)),
            }
        )

    @staticmethod
    def _to_weekly(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """
        Resample a daily DataFrame to weekly grain (Monday anchored).

        Numeric columns are summed; the date column becomes week-start.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        weekly = df.resample("W-MON").sum()
        weekly.index.name = date_col
        return weekly.reset_index()
