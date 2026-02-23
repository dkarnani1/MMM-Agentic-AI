"""
connectors/ga4_connector.py — Google Analytics 4 Sales Connector

Fetches weekly revenue (sales) from Google Analytics 4 using the
Google Analytics Data API.

Authentication — two options (set in .env):

  Option A — Service Account (recommended for production):
    GOOGLE_APPLICATION_CREDENTIALS  Path to service account JSON key file
    GA4_PROPERTY_ID                  GA4 property ID (numeric, e.g. "123456789")

  Option B — API Key (limited, read-only):
    GA4_API_KEY      API key with Analytics Data API enabled
    GA4_PROPERTY_ID  GA4 property ID

Install SDK:
    pip install google-analytics-data>=0.18.0

Metric used:
    purchaseRevenue — Total purchase revenue from ecommerce events.
    Divide by 1000 to convert to $000s.

Documentation:
    https://developers.google.com/analytics/devguides/reporting/data/v1

Note:
    When credentials are absent, fetch_to_dataframe() returns a zero-sales
    DataFrame so the pipeline can be tested without API access.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import numpy as np

from connectors.base import BaseConnector


class GA4Connector(BaseConnector):
    """
    Pulls weekly ecommerce revenue from Google Analytics 4.

    Usage:
        connector = GA4Connector()
        df = connector.fetch_to_dataframe(date(2023, 1, 1), date(2023, 12, 31))

    Returns:
        Weekly DataFrame with columns:
            date, sales, tv_spend (=0), radio_spend (=0),
            digital_spend (=0), print_spend (=0)
        (spend columns are always 0 — combine with spend connectors)
    """

    ENV_VARS_OPTION_A = ["GOOGLE_APPLICATION_CREDENTIALS", "GA4_PROPERTY_ID"]
    ENV_VARS_OPTION_B = ["GA4_API_KEY", "GA4_PROPERTY_ID"]

    def __init__(self) -> None:
        self._property_id = os.getenv("GA4_PROPERTY_ID")
        self._credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self._api_key     = os.getenv("GA4_API_KEY")

    def is_configured(self) -> bool:
        """Return True if at least one auth option is fully configured."""
        option_a = all(os.getenv(v) for v in self.ENV_VARS_OPTION_A)
        option_b = all(os.getenv(v) for v in self.ENV_VARS_OPTION_B)
        return option_a or option_b

    def fetch_to_dataframe(self, start: date, end: date) -> pd.DataFrame:
        """
        Fetch weekly GA4 purchase revenue.

        Falls back to a zero-sales DataFrame when credentials are absent.
        """
        if not self.is_configured():
            return self._fallback_with_notice(start, end)

        try:
            return self._fetch_live(start, end)
        except Exception as exc:
            print(f"[GA4Connector] API error: {exc}. Using fallback.")
            return self._fallback_synthetic(start, end)

    # == Private ============================================================

    def _fetch_live(self, start: date, end: date) -> pd.DataFrame:
        """
        Execute a GA4 Data API RunReport and aggregate to weekly grain.
        """
        from google.analytics.data_v1beta import BetaAnalyticsDataClient  # type: ignore
        from google.analytics.data_v1beta.types import (                   # type: ignore
            RunReportRequest, DateRange, Dimension, Metric,
        )

        # Service account auth uses GOOGLE_APPLICATION_CREDENTIALS automatically
        client = BetaAnalyticsDataClient()

        request = RunReportRequest(
            property=f"properties/{self._property_id}",
            dimensions=[Dimension(name="date")],
            metrics=[Metric(name="purchaseRevenue")],
            date_ranges=[
                DateRange(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                )
            ],
        )

        response = client.run_report(request)

        rows = []
        for row in response.rows:
            raw_date  = row.dimension_values[0].value   # "YYYYMMDD"
            revenue   = float(row.metric_values[0].value)
            rows.append(
                {
                    "date":  pd.to_datetime(raw_date, format="%Y%m%d"),
                    "sales": revenue / 1000.0,   # USD -> $000s
                }
            )

        if not rows:
            return self._fallback_synthetic(start, end)

        daily = pd.DataFrame(rows)
        daily["tv_spend"]      = 0.0
        daily["radio_spend"]   = 0.0
        daily["digital_spend"] = 0.0
        daily["print_spend"]   = 0.0

        weekly = self._to_weekly(daily)
        return weekly[["date", "sales", "tv_spend", "radio_spend",
                        "digital_spend", "print_spend"]]

    def _fallback_with_notice(self, start: date, end: date) -> pd.DataFrame:
        print(
            "[GA4Connector] Credentials not configured. "
            "Set GOOGLE_APPLICATION_CREDENTIALS (or GA4_API_KEY) and "
            "GA4_PROPERTY_ID in .env to enable live data. "
            "Returning zero-sales fallback."
        )
        return self._fallback_synthetic(start, end)
