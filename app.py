"""
app.py — MMM AI Dashboard (Streamlit Web UI)

Usage:
    streamlit run app.py

Features:
    * Upload your own CSV or use built-in synthetic data
    * 7 analysis tabs: ROI | Saturation | Contributions |
                       Budget Optimizer | Trends | Executive Summary | Q&A
    * Interactive budget optimiser with spend sliders
    * Natural-language Q&A chat interface
    * All 6 presentation charts displayed inline

Architecture:
    @st.cache_resource initialises the full agent pipeline once.
    All tabs share the same fitted ModelingAgent via st.session_state.
"""

from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MMM AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Imports from the MMM pipeline
# ─────────────────────────────────────────────────────────────────────────────
from agents import (
    DataIngestionAgent,
    DataValidationAgent,
    FeatureEngineeringAgent,
    ModelingAgent,
    InsightGenerationAgent,
    ResponseFormattingAgent,
    BudgetOptimizationAgent,
)
from interface.nlp_router import NLPRouter
from config.settings import MEDIA_CHANNELS, SPEND_COLS, CHANNEL_COLORS
from plotly_charts import PlotlyCharts


# ─────────────────────────────────────────────────────────────────────────────
# System Initialisation (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def build_system(csv_bytes: Optional[bytes] = None, _company: str = "") -> Dict[str, Any]:
    """
    Build the full MMM pipeline. Cached so it only runs on first load
    or when the data source changes.
    """
    ingestion = DataIngestionAgent()

    if csv_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name
        df_raw = ingestion.load_csv(tmp_path)
        os.unlink(tmp_path)
    else:
        df_raw = ingestion.load_synthetic()

    validator = DataValidationAgent()
    report    = validator.validate(df_raw)
    if not report["is_valid"]:
        raise ValueError("Data validation failed: " + "; ".join(report["errors"]))

    fe_agent    = FeatureEngineeringAgent()
    df_eng      = fe_agent.engineer_all_features(df_raw)

    model_agent = ModelingAgent()
    model_agent.fit(df_eng)

    insight_agent = InsightGenerationAgent(model_agent)
    formatter     = ResponseFormattingAgent()
    router        = NLPRouter(use_claude_api=True)

    return {
        "df":      df_eng,
        "model":   model_agent,
        "insight": insight_agent,
        "format":  formatter,
        "router":  router,
        "source":  ingestion.source,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chart_switcher(build_fn, options: dict, key: str) -> None:
    """
    Render an interactive Plotly chart with a chart-type toggle above it.

    build_fn : callable(chart_type: str) -> go.Figure | pd.DataFrame
    options  : {"Label": "type_key", ...}  — ordered dict shown as radio buttons
    key      : unique Streamlit widget key for this chart instance
    """
    selected   = st.radio(
        "View as",
        list(options.keys()),
        horizontal=True,
        key=f"sw_{key}",
        label_visibility="collapsed",
    )
    chart_type = options[selected]
    output     = build_fn(chart_type)
    if isinstance(output, pd.DataFrame):
        st.dataframe(output, width="stretch", hide_index=True)
    else:
        st.plotly_chart(output, width="stretch", key=f"pc_{key}")


def _insight_card(result: Dict) -> None:
    """Render a standard insight result dict as Streamlit components."""
    if "narrative" in result:
        st.markdown(result["narrative"])

    if "table_data" in result and result["table_data"]:
        df_table = pd.DataFrame(result["table_data"])
        st.dataframe(df_table, width="stretch", hide_index=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if result.get("key_insights"):
            st.markdown("**Key Insights**")
            for ins in result["key_insights"]:
                st.markdown(ins)
    with col_r:
        if result.get("actions"):
            st.markdown("**Recommended Actions**")
            for act in result["actions"]:
                st.markdown(act)

    if result.get("risks"):
        with st.expander("Risks & Caveats"):
            for r in result["risks"]:
                st.markdown(f"- {r}")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Optional[bytes]:
    """Render sidebar and return CSV bytes if a file was uploaded."""
    st.sidebar.title("MMM AI Dashboard")
    st.sidebar.markdown("*Marketing Mix Model — Executive Analytics*")
    st.sidebar.divider()

    data_source = st.sidebar.radio(
        "Data source",
        ["Synthetic (built-in)", "Upload CSV"],
        index=0,
    )

    csv_bytes = None
    if data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader(
            "Upload your MMM dataset",
            type=["csv"],
            help="Required columns: date, sales, tv_spend, radio_spend, digital_spend, print_spend",
        )
        if uploaded is not None:
            csv_bytes = uploaded.read()

    st.sidebar.divider()
    st.sidebar.markdown("**Required CSV columns**")
    st.sidebar.code(
        "date\nsales\ntv_spend\nradio_spend\ndigital_spend\nprint_spend",
        language="text",
    )
    st.sidebar.divider()
    if st.sidebar.button("Clear Cache & Rebuild", help="Force the model to re-initialise (use after changing settings.py)"):
        build_system.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("**About**")
    st.sidebar.markdown(
        "Built with Python · Powered by Claude · "
        "OLS regression with adstock & Hill-function saturation."
    )

    return csv_bytes


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — ROI Analysis
# ─────────────────────────────────────────────────────────────────────────────

def tab_roi(system: Dict) -> None:
    st.header("Channel ROI Analysis")
    st.caption("Return on investment: incremental sales generated per dollar of media spend.")

    result = system["insight"].roi_analysis()
    _insight_card(result)

    st.divider()
    pc = PlotlyCharts(system["model"])
    _chart_switcher(
        pc.roi_comparison,
        {"Horizontal Bar": "hbar", "Vertical Bar": "vbar", "Radar": "radar", "Table": "table"},
        key="roi",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Saturation
# ─────────────────────────────────────────────────────────────────────────────

def tab_saturation(system: Dict) -> None:
    st.header("Saturation & Diminishing Returns")
    st.caption(
        "How far each channel is along its response curve. "
        "Saturated channels have little headroom; under-invested channels do."
    )

    result = system["insight"].saturation_analysis()

    # Per-channel status cards
    if "channel_status" in result:
        cols = st.columns(len(MEDIA_CHANNELS))
        status_color = {"SATURATED": "🔴", "MODERATE": "🟡", "UNDER-INVESTED": "🟢"}
        for i, ch in enumerate(MEDIA_CHANNELS):
            ch_data = result["channel_status"].get(ch, {})
            icon    = status_color.get(ch_data.get("status", ""), "⚪")
            with cols[i]:
                st.metric(
                    label=f"{icon} {ch}",
                    value=f"{ch_data.get('saturation_pct', 0):.1f}%",
                    help=ch_data.get("status", ""),
                )

    _insight_card(result)

    st.divider()
    pc = PlotlyCharts(system["model"])
    _chart_switcher(
        pc.response_curves,
        {"Line Grid": "line", "Overlay": "scatter", "Table": "table"},
        key="saturation",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Contributions
# ─────────────────────────────────────────────────────────────────────────────

def tab_contributions(system: Dict) -> None:
    st.header("Sales Contribution Decomposition")
    st.caption("How much of total sales each media channel (and the organic base) explains.")

    result = system["insight"].contribution_analysis()

    if "pct" in result:
        cols = st.columns(len(result["pct"]))
        for i, (label, pct) in enumerate(result["pct"].items()):
            with cols[i]:
                st.metric(label=label, value=f"{pct:.1f}%")

    _insight_card(result)

    st.divider()
    pc = PlotlyCharts(system["model"])
    st.markdown("**Contribution Decomposition**")
    _chart_switcher(
        pc.channel_contributions,
        {"Stacked Area": "stacked_area", "Stacked Bar": "stacked_bar", "Lines": "line", "Table": "table"},
        key="contribs",
    )
    st.markdown("**Sales Breakdown**")
    _chart_switcher(
        pc.decomposition_pie,
        {"Donut": "donut", "Pie": "pie", "Bar": "bar", "Table": "table"},
        key="decomp",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Budget Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def tab_budget_optimizer(system: Dict) -> None:
    st.header("Budget Optimiser")
    st.caption(
        "scipy.optimize finds the spend allocation that maximises predicted sales "
        "within your total budget constraint and per-channel bounds."
    )

    model = system["model"]
    current_alloc = {
        ch: float(model.df[SPEND_COLS[ch]].mean())
        for ch in MEDIA_CHANNELS
    }
    current_total = sum(current_alloc.values())

    # ── Controls ────────────────────────────────────────────────────────────
    st.subheader("Budget Settings")
    col_budget, col_gap = st.columns([2, 3])
    with col_budget:
        total_budget = st.number_input(
            "Total weekly budget ($000s)",
            min_value=1.0,
            max_value=float(current_total * 5),
            value=float(round(current_total, 1)),
            step=50.0,
            help="Weekly budget to allocate optimally across channels.",
        )

    st.subheader("Per-Channel Spend Bounds ($000s / week)")
    min_alloc: Dict[str, float] = {}
    max_alloc: Dict[str, float] = {}
    bound_cols = st.columns(len(MEDIA_CHANNELS))

    for i, ch in enumerate(MEDIA_CHANNELS):
        cur = current_alloc[ch]
        with bound_cols[i]:
            st.markdown(f"**{ch}** *(current: ${cur:.0f}k)*")
            min_alloc[ch] = st.slider(
                f"Min {ch}",
                min_value=0.0,
                max_value=float(cur * 2),
                value=0.0,
                step=10.0,
                label_visibility="collapsed",
                key=f"min_{ch}",
            )
            max_alloc[ch] = st.slider(
                f"Max {ch}",
                min_value=float(cur * 0.5),
                max_value=float(cur * 4),
                value=float(cur * 3),
                step=50.0,
                label_visibility="collapsed",
                key=f"max_{ch}",
            )
            st.caption(f"Floor: ${min_alloc[ch]:.0f}k | Cap: ${max_alloc[ch]:.0f}k")

    # ── Run optimisation ────────────────────────────────────────────────────
    if st.button("Optimise Budget Allocation", type="primary", use_container_width=True):
        with st.spinner("Running scipy.optimize (SLSQP)..."):
            opt_agent = BudgetOptimizationAgent(model)
            result    = opt_agent.optimize(
                total_budget=total_budget,
                min_alloc=min_alloc,
                max_alloc=max_alloc,
            )

        # Headline metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Budget (weekly $k)", f"${total_budget:,.0f}k"
        )
        m2.metric(
            "Current Sales Lift (wk)",
            f"${result['current_sales_lift']:,.1f}k",
        )
        m3.metric(
            "Optimal Sales Lift (wk)",
            f"${result['optimal_sales_lift']:,.1f}k",
            delta=f"+{result['improvement_pct']:.1f}%",
        )
        m4.metric(
            "Annual Revenue Gain",
            f"${result['annual_improvement']:,.0f}k",
        )

        # Allocation table
        st.subheader("Recommended Reallocation")
        df_alloc = pd.DataFrame(result["table_data"])
        st.dataframe(df_alloc, width="stretch", hide_index=True)

        # Side-by-side bar chart
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Current vs Optimal Spend**")
            df_compare = pd.DataFrame(
                {
                    "Channel": MEDIA_CHANNELS,
                    "Current ($k)":  [round(result["current_allocation"][ch], 1) for ch in MEDIA_CHANNELS],
                    "Optimal ($k)":  [round(result["optimal_allocation"][ch], 1) for ch in MEDIA_CHANNELS],
                }
            )
            st.dataframe(df_compare, width="stretch", hide_index=True)

        with col_r:
            try:
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Bar(
                        name="Current",
                        x=MEDIA_CHANNELS,
                        y=[result["current_allocation"][ch] for ch in MEDIA_CHANNELS],
                        marker_color="#607D8B",
                    ),
                    go.Bar(
                        name="Optimal",
                        x=MEDIA_CHANNELS,
                        y=[result["optimal_allocation"][ch] for ch in MEDIA_CHANNELS],
                        marker_color="#4CAF50",
                    ),
                ])
                fig.update_layout(
                    barmode="group",
                    title="Weekly Spend: Current vs Optimal ($000s)",
                    yaxis_title="$000s / week",
                    height=350,
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig, width="stretch")
            except ImportError:
                pass

        # Insights and actions
        _insight_card(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Trends
# ─────────────────────────────────────────────────────────────────────────────

def tab_trends(system: Dict) -> None:
    st.header("Sales & Spend Trends")
    st.caption("Historical performance, quarterly breakdowns, and seasonality patterns.")

    period = st.selectbox(
        "Filter by quarter (optional)",
        options=["All periods", "Q1", "Q2", "Q3", "Q4"],
        index=0,
    )
    qtr = None if period == "All periods" else period

    result = system["insight"].trend_analysis(period=qtr)

    if "quarterly" in result and result["quarterly"]:
        st.subheader("Quarterly Sales Rollup")
        qdata = result["quarterly"]
        # quarterly is {component: {quarter_label: value}} — pivot to quarter rows
        if isinstance(next(iter(qdata.values())), dict):
            quarters = list(next(iter(qdata.values())).keys())
            rows = []
            for q in quarters:
                row = {"Quarter": q}
                for component, series in qdata.items():
                    row[f"{component} ($k)"] = f"${series.get(q, 0):,.0f}k"
                rows.append(row)
            df_qtr = pd.DataFrame(rows)
        else:
            df_qtr = pd.DataFrame(
                [{"Quarter": k, "Sales ($k)": f"${v:,.0f}k"} for k, v in qdata.items()]
            )
        st.dataframe(df_qtr, width="stretch", hide_index=True)

    _insight_card(result)

    st.divider()
    pc = PlotlyCharts(system["model"])
    st.markdown("**Sales & Media Spend Timeline**")
    _chart_switcher(
        pc.sales_spend_trend,
        {"Combo (Bar+Line)": "combo", "Lines": "line", "Spend Bars": "bar", "Table": "table"},
        key="trend",
    )
    st.markdown("**Budget Efficiency Matrix**")
    _chart_switcher(
        pc.budget_efficiency,
        {"Bubble": "bubble", "Bar": "bar", "Table": "table"},
        key="budeff",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Executive Summary
# ─────────────────────────────────────────────────────────────────────────────

def tab_executive(system: Dict) -> None:
    st.header("Executive Summary")
    st.caption("Full-picture briefing: ROI, saturation, contributions, and top recommendations.")

    result     = system["insight"].executive_summary()
    fit        = system["model"].get_fit_quality()

    # Model quality strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model R²",         str(fit["R²"]))
    m2.metric("Adj. R²",          str(fit["Adj. R²"]))
    m3.metric("MAPE",             f"{fit['MAPE (%)']:.1f}%")
    m4.metric("Weeks of data",    str(fit["N observations"]))

    st.divider()
    _insight_card(result)

    # All charts (interactive)
    st.divider()
    st.subheader("Presentation Charts")
    pc = PlotlyCharts(system["model"])
    exec_charts = [
        (
            "Sales & Media Spend Timeline",
            pc.sales_spend_trend,
            {"Combo (Bar+Line)": "combo", "Lines": "line", "Spend Bars": "bar", "Table": "table"},
            "ex_trend",
        ),
        (
            "Channel Contribution Decomposition",
            pc.channel_contributions,
            {"Stacked Area": "stacked_area", "Stacked Bar": "stacked_bar", "Lines": "line", "Table": "table"},
            "ex_contribs",
        ),
        (
            "Channel ROI Comparison",
            pc.roi_comparison,
            {"Horizontal Bar": "hbar", "Vertical Bar": "vbar", "Radar": "radar", "Table": "table"},
            "ex_roi",
        ),
        (
            "Saturation / Response Curves",
            pc.response_curves,
            {"Line Grid": "line", "Overlay": "scatter", "Table": "table"},
            "ex_sat",
        ),
        (
            "Budget Efficiency Matrix",
            pc.budget_efficiency,
            {"Bubble": "bubble", "Bar": "bar", "Table": "table"},
            "ex_budeff",
        ),
        (
            "Sales Decomposition",
            pc.decomposition_pie,
            {"Donut": "donut", "Pie": "pie", "Bar": "bar", "Table": "table"},
            "ex_decomp",
        ),
    ]
    for i in range(0, len(exec_charts), 2):
        col_l, col_r = st.columns(2)
        with col_l:
            label, fn, opts, chart_key = exec_charts[i]
            st.markdown(f"**{label}**")
            _chart_switcher(fn, opts, key=chart_key)
        if i + 1 < len(exec_charts):
            with col_r:
                label, fn, opts, chart_key = exec_charts[i + 1]
                st.markdown(f"**{label}**")
                _chart_switcher(fn, opts, key=chart_key)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 7 — Q&A Chat
# ─────────────────────────────────────────────────────────────────────────────

def tab_qa(system: Dict) -> None:
    st.header("Ask the MMM AI")
    st.caption(
        "Ask questions in plain English. The NLP router classifies intent and "
        "returns the relevant analysis."
    )

    # ── Session-state defaults ───────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "qa_input" not in st.session_state:
        st.session_state["qa_input"] = ""
    # Consume a deferred clear-request BEFORE the text_area widget is rendered.
    # (Streamlit forbids writing a widget's key after the widget is instantiated.)
    if st.session_state.pop("_qa_clear", False):
        st.session_state["qa_input"] = ""

    # ── chart_specs options ──────────────────────────────────────────────────
    _ROI_OPTS    = {"Horizontal Bar": "hbar", "Vertical Bar": "vbar", "Radar": "radar", "Table": "table"}
    _SAT_OPTS    = {"Line Grid": "line", "Overlay": "scatter", "Table": "table"}
    _CONT_OPTS   = {"Stacked Area": "stacked_area", "Stacked Bar": "stacked_bar", "Lines": "line", "Table": "table"}
    _DECOMP_OPTS = {"Donut": "donut", "Pie": "pie", "Bar": "bar", "Table": "table"}
    _TREND_OPTS  = {"Combo (Bar+Line)": "combo", "Lines": "line", "Spend Bars": "bar", "Table": "table"}
    _EFF_OPTS    = {"Bubble": "bubble", "Bar": "bar", "Table": "table"}

    pc = PlotlyCharts(system["model"])

    def _render_chat_entry(entry: Dict, key_prefix: str) -> None:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(f"**Intent detected:** `{entry['intent']}`")
            if "table_data" in entry["result"] and entry["result"]["table_data"]:
                st.dataframe(
                    pd.DataFrame(entry["result"]["table_data"]),
                    width="stretch",
                    hide_index=True,
                )
            st.markdown(entry["result"].get("narrative", ""))
            if entry["result"].get("key_insights"):
                for ins in entry["result"]["key_insights"]:
                    st.markdown(ins)
            if entry["result"].get("actions"):
                st.markdown("**Actions:**")
                for act in entry["result"]["actions"]:
                    st.markdown(act)
            for label, method_name, opts in entry.get("chart_specs", []):
                st.markdown(f"**{label}**")
                _chart_switcher(
                    getattr(pc, method_name), opts,
                    key=f"{key_prefix}_{method_name}",
                )

    # ── Example questions (clickable buttons → populate input) ───────────────
    example_qs = [
        "What is the ROI of TV vs Digital?",
        "Which channels are saturated?",
        "What if we cut TV by 20%?",
        "What drove Q4 performance?",
        "What's the optimal budget allocation?",
        "Give me an executive summary",
    ]
    st.markdown("**Example questions** — click to populate the box below:")
    btn_cols = st.columns(3)
    for i, q in enumerate(example_qs):
        if btn_cols[i % 3].button(q, key=f"eq_{i}", use_container_width=True):
            st.session_state["qa_input"] = q

    # ── Question input (text area + Ask button) ──────────────────────────────
    st.divider()
    question_text = st.text_area(
        "Your question",
        key="qa_input",
        height=80,
        placeholder="Ask a marketing mix question…",
        label_visibility="collapsed",
    )
    col_ask, col_clear, _ = st.columns([1, 1, 5])
    ask_clicked   = col_ask.button("Ask", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear history", use_container_width=True)

    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state["_qa_clear"]  = True
        st.rerun()

    if ask_clicked and question_text.strip():
        question = question_text.strip()
        router   = system["router"]
        insight  = system["insight"]
        model    = system["model"]

        intent, entities = router.parse(question)

        # Clarifying question if entities incomplete
        clarify = router.get_clarifying_question(intent, entities)
        if clarify:
            st.info(clarify)
        else:
            # Dispatch to insight method + build chart_specs for interactive replay
            chart_specs: list = []
            if intent == "roi_analysis":
                result      = insight.roi_analysis()
                chart_specs = [("ROI Comparison", "roi_comparison", _ROI_OPTS)]
            elif intent == "saturation":
                result      = insight.saturation_analysis()
                chart_specs = [("Response Curves", "response_curves", _SAT_OPTS)]
            elif intent == "contribution":
                result      = insight.contribution_analysis()
                chart_specs = [
                    ("Channel Contributions", "channel_contributions", _CONT_OPTS),
                    ("Sales Breakdown",       "decomposition_pie",     _DECOMP_OPTS),
                ]
            elif intent == "budget_scenario":
                channel = entities.get("channel", "Digital")
                pct     = entities.get("pct", -0.20)
                result  = insight.budget_scenario(channel, pct)
            elif intent == "trend":
                result      = insight.trend_analysis(period=entities.get("quarter"))
                chart_specs = [
                    ("Sales & Spend Trend", "sales_spend_trend", _TREND_OPTS),
                    ("Budget Efficiency",   "budget_efficiency", _EFF_OPTS),
                ]
            elif intent == "budget_optimize":
                total_budget = entities.get(
                    "total_budget",
                    sum(model.df[SPEND_COLS[ch]].mean() for ch in MEDIA_CHANNELS),
                )
                opt    = BudgetOptimizationAgent(model)
                result = opt.optimize(total_budget=total_budget)
            else:
                result      = insight.executive_summary()
                chart_specs = [
                    ("Sales & Spend Trend",    "sales_spend_trend",     _TREND_OPTS),
                    ("Channel Contributions",  "channel_contributions", _CONT_OPTS),
                    ("ROI Comparison",         "roi_comparison",        _ROI_OPTS),
                    ("Response Curves",        "response_curves",       _SAT_OPTS),
                    ("Budget Efficiency",      "budget_efficiency",     _EFF_OPTS),
                    ("Sales Decomposition",    "decomposition_pie",     _DECOMP_OPTS),
                ]

            st.session_state.chat_history.append(
                {"question": question, "intent": intent, "result": result,
                 "chart_specs": chart_specs}
            )
            # Signal the next run to clear the input box (must happen before
            # the text_area widget is instantiated, so we use a flag).
            st.session_state["_qa_clear"] = True
            st.rerun()

    # ── Chat history — most recent first ─────────────────────────────────────
    st.divider()
    history = st.session_state.chat_history
    for h_idx in range(len(history) - 1, -1, -1):
        _render_chat_entry(history[h_idx], key_prefix=f"qah{h_idx}")


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_bytes = render_sidebar()

    # Initialise / re-initialise system
    # Include COMPANY_NAME in cache_key so a name change auto-invalidates
    from config.settings import COMPANY_NAME
    cache_key = (hash(csv_bytes) if csv_bytes else "synthetic", COMPANY_NAME)
    if (
        "system" not in st.session_state
        or st.session_state.get("cache_key") != cache_key
    ):
        with st.spinner("Initialising MMM pipeline (this takes ~5 seconds)..."):
            try:
                system = build_system(csv_bytes, _company=COMPANY_NAME)
                st.session_state["system"]    = system
                st.session_state["cache_key"] = cache_key
                st.session_state.pop("init_error", None)
            except Exception as exc:
                st.session_state["init_error"] = str(exc)

    if "init_error" in st.session_state:
        st.error(f"Initialisation failed: {st.session_state['init_error']}")
        return

    system = st.session_state["system"]

    # Data source badge
    st.caption(f"Data source: **{system['source']}**")

    # Tabs
    tabs = st.tabs([
        "ROI Analysis",
        "Saturation",
        "Contributions",
        "Budget Optimizer",
        "Trends",
        "Executive Summary",
        "Q&A Chat",
    ])

    with tabs[0]:
        tab_roi(system)
    with tabs[1]:
        tab_saturation(system)
    with tabs[2]:
        tab_contributions(system)
    with tabs[3]:
        tab_budget_optimizer(system)
    with tabs[4]:
        tab_trends(system)
    with tabs[5]:
        tab_executive(system)
    with tabs[6]:
        tab_qa(system)


if __name__ == "__main__":
    main()
