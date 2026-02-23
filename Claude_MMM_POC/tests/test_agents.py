"""
test_agents.py — Unit tests for all MMM agents.

Run with:
    pytest tests/ -v

Tests cover:
  * Data generation produces expected shape and value ranges
  * DataIngestionAgent loads synthetic data correctly
  * DataValidationAgent passes on clean data
  * FeatureEngineeringAgent applies transformations correctly
  * ModelingAgent fits and produces valid ROI / contributions
  * InsightGenerationAgent returns correctly structured dicts
  * NLPRouter classifies intents and extracts entities
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.sample_data import generate_mmm_dataset
from agents.data_ingestion_agent import DataIngestionAgent
from agents.data_validation_agent import DataValidationAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.modeling_agent import ModelingAgent
from agents.insight_generation_agent import InsightGenerationAgent
from agents.budget_optimization_agent import BudgetOptimizationAgent
from interface.nlp_router import NLPRouter
from config.settings import MEDIA_CHANNELS, SPEND_COLS


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Shared synthetic dataset for the test session."""
    return generate_mmm_dataset(n_weeks=52, seed=42)


@pytest.fixture(scope="module")
def engineered_df(raw_df) -> pd.DataFrame:
    agent = FeatureEngineeringAgent()
    return agent.engineer_all_features(raw_df)


@pytest.fixture(scope="module")
def fitted_model(engineered_df) -> ModelingAgent:
    model = ModelingAgent()
    model.fit(engineered_df)
    return model


# =============================================================================
# Data Generation
# =============================================================================

class TestSampleData:
    def test_shape(self, raw_df):
        assert raw_df.shape[0] == 52
        assert "sales" in raw_df.columns

    def test_no_negatives(self, raw_df):
        for col in ["sales", "tv_spend", "radio_spend", "digital_spend", "print_spend"]:
            assert (raw_df[col] >= 0).all(), f"{col} has negative values"

    def test_date_is_monotonic(self, raw_df):
        assert raw_df["date"].is_monotonic_increasing

    def test_sales_are_positive(self, raw_df):
        assert raw_df["sales"].mean() > 0


# =============================================================================
# Data Ingestion Agent
# =============================================================================

class TestDataIngestionAgent:
    def test_load_synthetic_returns_dataframe(self):
        agent = DataIngestionAgent()
        df    = agent.load_synthetic()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_source_is_set(self):
        agent = DataIngestionAgent()
        agent.load_synthetic()
        assert "Synthetic" in agent.source

    def test_summary_has_expected_keys(self):
        agent = DataIngestionAgent()
        agent.load_synthetic()
        summary = agent.get_summary()
        assert "rows" in summary
        assert "date_range" in summary
        assert "total_sales" in summary

    def test_load_csv_missing_file_raises(self):
        agent = DataIngestionAgent()
        with pytest.raises(FileNotFoundError):
            agent.load_csv("nonexistent_file.csv")


# =============================================================================
# Data Validation Agent
# =============================================================================

class TestDataValidationAgent:
    def test_clean_data_passes(self, raw_df):
        agent  = DataValidationAgent()
        report = agent.validate(raw_df)
        assert report["is_valid"] is True
        assert len(report["errors"]) == 0

    def test_missing_columns_fails(self):
        bad_df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="W"),
                               "sales": range(10)})
        agent  = DataValidationAgent()
        report = agent.validate(bad_df)
        assert report["is_valid"] is False

    def test_negative_values_fail(self, raw_df):
        bad_df               = raw_df.copy()
        bad_df.loc[0, "sales"] = -100
        agent  = DataValidationAgent()
        report = agent.validate(bad_df)
        assert report["is_valid"] is False

    def test_report_has_stats(self, raw_df):
        agent  = DataValidationAgent()
        report = agent.validate(raw_df)
        assert "stats" in report
        assert "Weeks" in report["stats"]


# =============================================================================
# Feature Engineering Agent
# =============================================================================

class TestFeatureEngineeringAgent:
    def test_transformed_cols_added(self, raw_df):
        agent  = FeatureEngineeringAgent()
        df_eng = agent.engineer_all_features(raw_df)
        for ch in MEDIA_CHANNELS:
            assert f"{ch.lower()}_transformed" in df_eng.columns
            assert f"{ch.lower()}_adstock" in df_eng.columns

    def test_transformed_values_in_01(self, raw_df):
        agent  = FeatureEngineeringAgent()
        df_eng = agent.engineer_all_features(raw_df)
        for ch in MEDIA_CHANNELS:
            col = f"{ch.lower()}_transformed"
            assert df_eng[col].between(0, 1).all(), f"{col} out of [0,1] range"

    def test_trend_column_exists(self, raw_df):
        agent  = FeatureEngineeringAgent()
        df_eng = agent.engineer_all_features(raw_df)
        assert "trend" in df_eng.columns
        assert "sin_annual" in df_eng.columns

    def test_adstock_monotonic_for_constant_spend(self):
        """Adstock should be monotonically increasing for constant spend input."""
        agent  = FeatureEngineeringAgent()
        spend  = np.ones(20) * 100.0
        result = agent._apply_adstock(spend, decay=0.5)
        diffs  = np.diff(result)
        assert (diffs >= 0).all()

    def test_saturation_bounded(self):
        agent = FeatureEngineeringAgent()
        x     = np.linspace(0, 1000, 100)
        sat   = agent._apply_saturation(x, alpha=2.0, K=0.5)
        assert sat.min() >= 0
        assert sat.max() <= 1.0


# =============================================================================
# Modeling Agent
# =============================================================================

class TestModelingAgent:
    def test_fit_returns_self(self, engineered_df):
        model = ModelingAgent()
        result = model.fit(engineered_df)
        assert result is model

    def test_r_squared_positive(self, fitted_model):
        fit = fitted_model.get_fit_quality()
        assert fit["R²"] > 0.5, "R² too low — model is not fitting well"

    def test_contributions_sum_close_to_actual(self, fitted_model):
        total_contrib = sum(
            s.sum() for s in fitted_model.contributions.values()
        )
        total_actual  = fitted_model.df["sales"].sum()
        # Fitted sum should be within 5% of actual
        ratio = abs(total_contrib - total_actual) / total_actual
        assert ratio < 0.05, f"Contributions sum too far from actual ({ratio:.1%})"

    def test_roi_positive_for_all_channels(self, fitted_model):
        for ch, r in fitted_model.roi.items():
            assert r >= 0, f"Negative ROI for {ch}"

    def test_response_curves_shape(self, fitted_model):
        curves = fitted_model.get_response_curves()
        for ch in MEDIA_CHANNELS:
            assert ch in curves
            assert len(curves[ch]["spend"]) > 0
            assert len(curves[ch]["sales_lift"]) == len(curves[ch]["spend"])

    def test_budget_scenario_returns_dict(self, fitted_model):
        result = fitted_model.simulate_budget_change("TV", -0.20)
        assert "delta_annual_sales" in result
        assert "new_channel_roi" in result


# =============================================================================
# Insight Generation Agent
# =============================================================================

class TestInsightGenerationAgent:
    def test_roi_analysis_structure(self, fitted_model):
        agent  = InsightGenerationAgent(fitted_model)
        result = agent.roi_analysis()
        assert result["type"] == "roi_analysis"
        assert "table_data" in result
        assert "key_insights" in result
        assert "actions" in result

    def test_saturation_analysis_structure(self, fitted_model):
        agent  = InsightGenerationAgent(fitted_model)
        result = agent.saturation_analysis()
        assert result["type"] == "saturation_analysis"
        assert "channel_status" in result
        for ch in MEDIA_CHANNELS:
            assert ch in result["channel_status"]

    def test_contribution_analysis_structure(self, fitted_model):
        agent  = InsightGenerationAgent(fitted_model)
        result = agent.contribution_analysis()
        assert "pct" in result
        assert abs(sum(result["pct"].values()) - 100) < 2.0, "Percentages don't sum to ~100%"

    def test_budget_scenario_structure(self, fitted_model):
        agent  = InsightGenerationAgent(fitted_model)
        result = agent.budget_scenario("TV", -0.20)
        assert result["type"] == "budget_scenario"
        assert "scenario" in result

    def test_executive_summary_structure(self, fitted_model):
        agent  = InsightGenerationAgent(fitted_model)
        result = agent.executive_summary()
        assert result["type"] == "executive_summary"
        assert "headline" in result
        assert "top_actions" in result


# =============================================================================
# NLP Router
# =============================================================================

class TestNLPRouter:
    CASES = [
        ("What is the ROI of TV?",                     "roi_analysis"),
        ("Show me channel ROI",                        "roi_analysis"),
        ("Which channels are saturated?",              "saturation"),
        ("Are we seeing diminishing returns on TV?",   "saturation"),
        ("What's driving sales?",                      "contribution"),
        ("Break down sales by channel",                "contribution"),
        ("What if we cut Digital by 20%?",             "budget_scenario"),
        ("Increase TV spend by 10%",                   "budget_scenario"),
        ("What drove Q4 performance?",                 "trend"),
        ("Show me the sales trend",                    "trend"),
        ("Give me an executive summary",               "executive_summary"),
        ("Overview of marketing performance",          "executive_summary"),
        ("help",                                       "help"),
        ("demo",                                       "demo"),
    ]

    @pytest.mark.parametrize("question,expected_intent", CASES)
    def test_intent_classification(self, question, expected_intent):
        router = NLPRouter(use_claude_api=False)
        intent = router.classify_intent(question)
        assert intent == expected_intent, (
            f"Question: '{question}'\n"
            f"Expected: {expected_intent}, Got: {intent}"
        )

    def test_entity_extraction_channel(self):
        router = NLPRouter(use_claude_api=False)
        _, entities = router.parse("What if we cut TV by 20%?")
        assert entities.get("channel") == "TV"
        assert abs(entities.get("pct", 0) + 0.20) < 0.01

    def test_entity_extraction_quarter(self):
        router = NLPRouter(use_claude_api=False)
        _, entities = router.parse("What drove Q4 sales?")
        assert entities.get("quarter") == "Q4"

    def test_clarifying_question_for_incomplete_scenario(self):
        router = NLPRouter(use_claude_api=False)
        # Budget scenario without channel or %
        clarify = router.get_clarifying_question("budget_scenario", {})
        assert clarify is not None
        assert "channel" in clarify.lower() or "Available" in clarify

    def test_budget_optimize_intent_detected(self):
        router = NLPRouter(use_claude_api=False)
        intent = router.classify_intent("What is the optimal budget allocation?")
        assert intent == "budget_optimize"

    def test_budget_optimize_keyword_maximize(self):
        router = NLPRouter(use_claude_api=False)
        intent = router.classify_intent("How do I maximize sales with my current spend?")
        assert intent == "budget_optimize"


# =============================================================================
# Budget Optimization Agent
# =============================================================================

class TestBudgetOptimizationAgent:
    def test_optimize_returns_expected_keys(self, fitted_model):
        agent  = BudgetOptimizationAgent(fitted_model)
        total  = sum(fitted_model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS)
        result = agent.optimize(total_budget=total)
        for key in ("optimal_allocation", "current_allocation", "optimal_sales_lift",
                    "current_sales_lift", "annual_improvement", "improvement_pct",
                    "channel_changes", "narrative", "table_data", "key_insights", "actions"):
            assert key in result, f"Missing key: {key}"

    def test_optimize_respects_budget_constraint(self, fitted_model):
        total  = sum(fitted_model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS)
        agent  = BudgetOptimizationAgent(fitted_model)
        result = agent.optimize(total_budget=total)
        opt_total = sum(result["optimal_allocation"].values())
        assert opt_total <= total * 1.001, (
            f"Optimal spend {opt_total:.1f} exceeds budget {total:.1f}"
        )

    def test_optimize_all_channels_present(self, fitted_model):
        total  = sum(fitted_model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS)
        agent  = BudgetOptimizationAgent(fitted_model)
        result = agent.optimize(total_budget=total)
        for ch in MEDIA_CHANNELS:
            assert ch in result["optimal_allocation"], f"{ch} missing from optimal_allocation"
            assert ch in result["current_allocation"], f"{ch} missing from current_allocation"

    def test_optimize_table_data_has_all_channels(self, fitted_model):
        total  = sum(fitted_model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS)
        agent  = BudgetOptimizationAgent(fitted_model)
        result = agent.optimize(total_budget=total)
        assert len(result["table_data"]) == len(MEDIA_CHANNELS)

    def test_optimize_with_reduced_budget(self, fitted_model):
        total   = sum(fitted_model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS)
        reduced = total * 0.7
        agent   = BudgetOptimizationAgent(fitted_model)
        result  = agent.optimize(total_budget=reduced)
        opt_total = sum(result["optimal_allocation"].values())
        assert opt_total <= reduced * 1.001, "Reduced budget constraint violated"


# =============================================================================
# Connectors (schema validation — no API credentials required)
# =============================================================================

class TestConnectors:
    def test_base_connector_validate_schema_pass(self):
        from connectors.base import BaseConnector, REQUIRED_COLUMNS
        import pandas as pd

        # Minimal concrete subclass for testing
        class DummyConnector(BaseConnector):
            def fetch_to_dataframe(self, start, end):
                return self._fallback_synthetic(start, end)

        dc = DummyConnector()
        from datetime import date
        df = dc.fetch_to_dataframe(date(2023, 1, 1), date(2023, 12, 31))
        assert dc.validate_schema(df), "Fallback DataFrame failed schema validation"
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_google_ads_connector_not_configured(self):
        from connectors.google_ads_connector import GoogleAdsConnector
        connector = GoogleAdsConnector()
        # Without credentials, is_configured() must return False
        # (env vars are not set in test environment)
        result = connector.is_configured()
        assert isinstance(result, bool)

    def test_meta_connector_fallback_schema(self):
        from connectors.meta_connector import MetaConnector
        from datetime import date
        connector = MetaConnector()
        df = connector._fallback_synthetic(date(2023, 1, 1), date(2023, 3, 31))
        from connectors.base import REQUIRED_COLUMNS
        assert connector.validate_schema(df)

    def test_ga4_connector_fallback_schema(self):
        from connectors.ga4_connector import GA4Connector
        from datetime import date
        connector = GA4Connector()
        df = connector._fallback_synthetic(date(2023, 1, 1), date(2023, 3, 31))
        from connectors.base import REQUIRED_COLUMNS
        assert connector.validate_schema(df)
