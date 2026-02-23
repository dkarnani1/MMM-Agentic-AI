"""
connectors/google_ads_connector.py — Google Ads API Connector

Fetches weekly media spend from Google Ads and maps campaign types
to MMM channel buckets (TV, Radio, Digital, Print).

Authentication (set in .env):
    GOOGLE_ADS_DEVELOPER_TOKEN   — Developer token from Google Ads API Centre
    GOOGLE_ADS_CLIENT_ID         — OAuth2 client ID
    GOOGLE_ADS_CLIENT_SECRET     — OAuth2 client secret
    GOOGLE_ADS_REFRESH_TOKEN     — OAuth2 refresh token (from OAuth flow)
    GOOGLE_ADS_CUSTOMER_ID       — 10-digit customer/account ID (no dashes)

Install SDK:
    pip install google-ads>=24.0.0

Channel Mapping (customise via CAMPAIGN_TYPE_MAP):
    SEARCH        -> Digital
    DISPLAY       -> Digital
    VIDEO         -> TV (YouTube awareness campaigns)
    SHOPPING      -> Digital
    SMART         -> Digital
    PERFORMANCE_MAX -> Digital

Documentation:
    https://developers.google.com/google-ads/api/docs/start

Note:
    When credentials are absent, fetch_to_dataframe() returns a zero-spend
    DataFrame so the pipeline can be tested without API access.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict, Optional

import pandas as pd
import numpy as np

from connectors.base import BaseConnector


# Maps Google Ads campaign type enum -> MMM channel bucket
CAMPAIGN_TYPE_MAP: Dict[str, str] = {
    "SEARCH":          "digital_spend",
    "DISPLAY":         "digital_spend",
    "VIDEO":           "tv_spend",       # YouTube brand awareness
    "SHOPPING":        "digital_spend",
    "SMART":           "digital_spend",
    "PERFORMANCE_MAX": "digital_spend",
    "APP":             "digital_spend",
    "DISCOVERY":       "digital_spend",
    "HOTEL":           "digital_spend",
    "LOCAL":           "digital_spend",
    "LOCAL_SERVICES":  "digital_spend",
}


class GoogleAdsConnector(BaseConnector):
    """
    Pulls weekly spend by campaign type from the Google Ads API.

    Usage:
        connector = GoogleAdsConnector()
        df = connector.fetch_to_dataframe(date(2023, 1, 1), date(2023, 12, 31))

    Returns:
        Weekly DataFrame with columns:
            date, sales (=0), tv_spend, radio_spend, digital_spend, print_spend
        (sales is always 0 — fetch sales from GA4Connector instead)
    """

    ENV_VARS = [
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET",
        "GOOGLE_ADS_REFRESH_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID",
    ]

    def __init__(self) -> None:
        self._customer_id: Optional[str] = os.getenv("GOOGLE_ADS_CUSTOMER_ID")

    def is_configured(self) -> bool:
        """Return True if all required environment variables are set."""
        return all(os.getenv(v) for v in self.ENV_VARS)

    def fetch_to_dataframe(self, start: date, end: date) -> pd.DataFrame:
        """
        Fetch weekly Google Ads spend by campaign type.

        Falls back to a zero-spend DataFrame when credentials are absent.
        """
        if not self.is_configured():
            return self._fallback_with_notice(start, end)

        try:
            return self._fetch_live(start, end)
        except Exception as exc:
            print(f"[GoogleAdsConnector] API error: {exc}. Using fallback.")
            return self._fallback_synthetic(start, end)

    # == Private ============================================================

    def _fetch_live(self, start: date, end: date) -> pd.DataFrame:
        """
        Execute a GAQL query and aggregate spend to weekly grain.

        GAQL: Google Ads Query Language — SQL-like query language.
        """
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore

        config = {
            "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN"),
            "client_id":       os.getenv("GOOGLE_ADS_CLIENT_ID"),
            "client_secret":   os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
            "refresh_token":   os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
            "use_proto_plus":  True,
        }
        client  = GoogleAdsClient.load_from_dict(config)
        service = client.get_service("GoogleAdsService")

        query = f"""
            SELECT
                campaign.advertising_channel_type,
                segments.date,
                metrics.cost_micros
            FROM campaign
            WHERE segments.date BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
              AND campaign.status = 'ENABLED'
        """

        response = service.search(
            customer_id=self._customer_id.replace("-", ""),
            query=query,
        )

        rows = []
        for row in response:
            rows.append(
                {
                    "date":          pd.to_datetime(row.segments.date),
                    "campaign_type": row.campaign.advertising_channel_type.name,
                    "cost_micros":   row.metrics.cost_micros,
                }
            )

        if not rows:
            return self._fallback_synthetic(start, end)

        df = pd.DataFrame(rows)
        # Convert micros -> $000s (1 micro = 1e-6 USD; divide by 1e9 for $000s)
        df["cost_k"] = df["cost_micros"] / 1_000_000_000.0

        # Map campaign type -> channel
        df["channel"] = df["campaign_type"].map(CAMPAIGN_TYPE_MAP).fillna("digital_spend")

        # Pivot to channel columns
        daily = df.pivot_table(
            index="date", columns="channel", values="cost_k", aggfunc="sum"
        ).fillna(0).reset_index()

        for col in ["tv_spend", "radio_spend", "digital_spend", "print_spend"]:
            if col not in daily.columns:
                daily[col] = 0.0

        daily["sales"] = 0.0

        weekly = self._to_weekly(daily)
        return weekly[["date", "sales", "tv_spend", "radio_spend",
                        "digital_spend", "print_spend"]]

    def _fallback_with_notice(self, start: date, end: date) -> pd.DataFrame:
        print(
            "[GoogleAdsConnector] Credentials not configured. "
            "Set GOOGLE_ADS_* env vars in .env to enable live data. "
            "Returning zero-spend fallback."
        )
        return self._fallback_synthetic(start, end)
