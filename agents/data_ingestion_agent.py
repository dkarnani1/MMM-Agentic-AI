"""
data_ingestion_agent.py — Agent 1: Data Ingestion

Responsibilities:
  * Load dataset from CSV or generate synthetic data
  * Return a standardised pandas DataFrame
  * Report data source and shape to the caller
"""

from __future__ import annotations

import pandas as pd
from datetime import date
from pathlib import Path
from typing import List, Optional

from config.settings import DATA_DIR, MEDIA_CHANNELS, SPEND_COLS


class DataIngestionAgent:
    """
    Loads and standardises the raw MMM dataset.

    Supports two data sources:
      1. CSV file  — user-supplied file with the expected column schema
      2. Synthetic — auto-generated 4-year dataset (default for POC)
    """

    REQUIRED_COLUMNS = ["date", "sales"] + list(SPEND_COLS.values())

    def __init__(self) -> None:
        self.source: str = "unknown"
        self.df: Optional[pd.DataFrame] = None

    # == Public Interface ====================================================

    def load_csv(self, filepath: str | Path) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            filepath: Path to the CSV.  Must contain at minimum:
                      date, sales, tv_spend, radio_spend,
                      digital_spend, print_spend

        Returns:
            Standardised DataFrame.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If required columns are missing.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path, parse_dates=["date"])
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Expected: {self.REQUIRED_COLUMNS}"
            )

        self.source = f"CSV ({path.name})"
        self.df = self._standardise(df)
        return self.df

    def load_synthetic(self) -> pd.DataFrame:
        """
        Generate and return the built-in synthetic MMM dataset.

        Returns:
            208-week synthetic DataFrame.
        """
        # Import here to avoid circular imports
        from data.sample_data import generate_mmm_dataset
        from config.settings import RANDOM_SEED, N_WEEKS

        df = generate_mmm_dataset(n_weeks=N_WEEKS, seed=RANDOM_SEED)
        self.source = "Synthetic (Google LightweightMMM-style)"
        self.df = self._standardise(df)
        return self.df

    def get_summary(self) -> dict:
        """Return a brief summary of the loaded dataset."""
        if self.df is None:
            return {"error": "No data loaded yet."}
        return {
            "source":      self.source,
            "rows":        len(self.df),
            "date_range":  f"{self.df['date'].min().date()} to {self.df['date'].max().date()}",
            "channels":    MEDIA_CHANNELS,
            "total_spend": {
                ch: f"${self.df[col].sum():,.0f}k"
                for ch, col in SPEND_COLS.items()
            },
            "total_sales": f"${self.df['sales'].sum():,.0f}k",
        }

    def load_live_data(
        self,
        sales_connector,
        spend_connectors: List,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Fetch and merge live data from API connectors.

        Args:
            sales_connector:   A GA4Connector (or any BaseConnector) that provides
                               the `sales` column.
            spend_connectors:  List of spend connectors (GoogleAdsConnector,
                               MetaConnector, etc.) whose spend columns are summed.
            start:             Start date for the fetch window.
            end:               End date for the fetch window.

        Returns:
            Standardised weekly DataFrame ready for the validation pipeline.
        """
        # Fetch sales
        df_sales = sales_connector.fetch_to_dataframe(start, end)[["date", "sales"]]

        # Fetch and merge spend from each connector
        spend_cols = list(SPEND_COLS.values())
        df_spend   = df_sales[["date"]].copy()
        for col in spend_cols:
            df_spend[col] = 0.0

        for connector in spend_connectors:
            df_c = connector.fetch_to_dataframe(start, end)
            df_c = df_c.set_index("date")
            for col in spend_cols:
                if col in df_c.columns:
                    df_c_aligned = df_c[col].reindex(df_spend["date"]).fillna(0.0).values
                    df_spend[col] = df_spend[col] + df_c_aligned

        df_merged = df_sales.merge(df_spend, on="date", how="left").fillna(0.0)

        self.source = f"Live API ({start} to {end})"
        self.df = self._standardise(df_merged)
        return self.df

    # == Private Helpers =====================================================

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by date, reset index, ensure numeric types."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        numeric_cols = ["sales"] + list(SPEND_COLS.values())
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
