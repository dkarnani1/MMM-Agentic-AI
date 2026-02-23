"""
connectors/meta_connector.py — Meta (Facebook / Instagram) Ads Connector

Fetches weekly media spend from the Meta Marketing API and maps campaign
objectives to MMM channel buckets.

Authentication (set in .env):
    META_ACCESS_TOKEN   — Long-lived System User token from Meta Business Manager
    META_AD_ACCOUNT_ID  — Ad account ID in format act_XXXXXXXXXX

Install SDK:
    pip install facebook-business>=19.0.0

Channel Mapping (customise via OBJECTIVE_MAP):
    BRAND_AWARENESS     -> tv_spend      (Upper-funnel brand building)
    REACH               -> tv_spend
    VIDEO_VIEWS         -> tv_spend
    TRAFFIC             -> digital_spend
    CONVERSIONS         -> digital_spend
    CATALOG_SALES       -> digital_spend
    STORE_TRAFFIC       -> digital_spend
    LEAD_GENERATION     -> digital_spend
    APP_INSTALLS        -> digital_spend
    MESSAGES            -> digital_spend
    POST_ENGAGEMENT     -> digital_spend

Documentation:
    https://developers.facebook.com/docs/marketing-api/

Note:
    When credentials are absent, fetch_to_dataframe() returns a zero-spend
    DataFrame so the pipeline can be tested without API access.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Dict

import pandas as pd
import numpy as np

from connectors.base import BaseConnector


# Maps Meta campaign objective -> MMM channel bucket
OBJECTIVE_MAP: Dict[str, str] = {
    "BRAND_AWARENESS":  "tv_spend",
    "REACH":            "tv_spend",
    "VIDEO_VIEWS":      "tv_spend",
    "TRAFFIC":          "digital_spend",
    "CONVERSIONS":      "digital_spend",
    "CATALOG_SALES":    "digital_spend",
    "STORE_TRAFFIC":    "digital_spend",
    "LEAD_GENERATION":  "digital_spend",
    "APP_INSTALLS":     "digital_spend",
    "MESSAGES":         "digital_spend",
    "POST_ENGAGEMENT":  "digital_spend",
    "LINK_CLICKS":      "digital_spend",
    "PRODUCT_CATALOG_SALES": "digital_spend",
    "OUTCOME_AWARENESS":     "tv_spend",
    "OUTCOME_TRAFFIC":       "digital_spend",
    "OUTCOME_ENGAGEMENT":    "digital_spend",
    "OUTCOME_LEADS":         "digital_spend",
    "OUTCOME_SALES":         "digital_spend",
    "OUTCOME_APP_PROMOTION": "digital_spend",
}


class MetaConnector(BaseConnector):
    """
    Pulls weekly spend by campaign objective from the Meta Marketing API.

    Usage:
        connector = MetaConnector()
        df = connector.fetch_to_dataframe(date(2023, 1, 1), date(2023, 12, 31))

    Returns:
        Weekly DataFrame with columns:
            date, sales (=0), tv_spend, radio_spend, digital_spend, print_spend
        (sales is always 0 — fetch sales from GA4Connector instead)
    """

    ENV_VARS = ["META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID"]

    def __init__(self) -> None:
        self._access_token  = os.getenv("META_ACCESS_TOKEN")
        self._ad_account_id = os.getenv("META_AD_ACCOUNT_ID")

    def is_configured(self) -> bool:
        """Return True if all required environment variables are set."""
        return all(os.getenv(v) for v in self.ENV_VARS)

    def fetch_to_dataframe(self, start: date, end: date) -> pd.DataFrame:
        """
        Fetch weekly Meta Ads spend by campaign objective.

        Falls back to a zero-spend DataFrame when credentials are absent.
        """
        if not self.is_configured():
            return self._fallback_with_notice(start, end)

        try:
            return self._fetch_live(start, end)
        except Exception as exc:
            print(f"[MetaConnector] API error: {exc}. Using fallback.")
            return self._fallback_synthetic(start, end)

    # == Private ============================================================

    def _fetch_live(self, start: date, end: date) -> pd.DataFrame:
        """
        Call the Meta Insights API and aggregate spend to weekly grain.
        """
        from facebook_business.adobjects.adaccount import AdAccount  # type: ignore
        from facebook_business.api import FacebookAdsApi               # type: ignore

        FacebookAdsApi.init(access_token=self._access_token)
        account = AdAccount(self._ad_account_id)

        fields = ["campaign_name", "objective", "spend", "date_start"]
        params = {
            "time_range": {
                "since": start.isoformat(),
                "until": end.isoformat(),
            },
            "level":       "campaign",
            "time_increment": 1,   # daily breakdown
            "limit":       500,
        }

        insights = account.get_insights(fields=fields, params=params)

        rows = []
        for record in insights:
            rows.append(
                {
                    "date":      pd.to_datetime(record["date_start"]),
                    "objective": record.get("objective", "CONVERSIONS"),
                    "spend_usd": float(record.get("spend", 0)),
                }
            )

        if not rows:
            return self._fallback_synthetic(start, end)

        df = pd.DataFrame(rows)
        # Convert USD -> $000s
        df["spend_k"] = df["spend_usd"] / 1000.0

        # Map objective -> channel
        df["channel"] = df["objective"].map(OBJECTIVE_MAP).fillna("digital_spend")

        daily = df.pivot_table(
            index="date", columns="channel", values="spend_k", aggfunc="sum"
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
            "[MetaConnector] Credentials not configured. "
            "Set META_ACCESS_TOKEN and META_AD_ACCOUNT_ID in .env to enable live data. "
            "Returning zero-spend fallback."
        )
        return self._fallback_synthetic(start, end)
