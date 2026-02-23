"""
connectors/ — Live Data Connectors for MMM Pipeline

Provides authenticated connectors to pull real media spend and sales data
from third-party APIs, returning DataFrames in the same schema as the
synthetic data so the rest of the pipeline is source-agnostic.

Usage:
    from connectors import GoogleAdsConnector, MetaConnector, GA4Connector
    from datetime import date

    ga4   = GA4Connector()
    gads  = GoogleAdsConnector()
    meta  = MetaConnector()

    ingestion = DataIngestionAgent()
    df = ingestion.load_live_data(
        sales_connector=ga4,
        spend_connectors=[gads, meta],
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
    )

Authentication:
    Set credentials in .env (see .env.example for required keys).
    Each connector falls back to a synthetic-structure DataFrame when
    credentials are absent — safe for local development and CI.
"""

from connectors.base import BaseConnector, REQUIRED_COLUMNS
from connectors.google_ads_connector import GoogleAdsConnector
from connectors.meta_connector import MetaConnector
from connectors.ga4_connector import GA4Connector

__all__ = [
    "BaseConnector",
    "REQUIRED_COLUMNS",
    "GoogleAdsConnector",
    "MetaConnector",
    "GA4Connector",
]
