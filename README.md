# Marketing Mix Model AI — Agentic Analytics POC
[Live App: MMM Agentic AI Streamlit](https://mmm-agentic-ai-kvhdwtb6qtgcr3hxufrwpc.streamlit.app/)

> **An AI-powered Marketing Mix Modeling system that answers executive questions in plain English, quantifies channel ROI, and generates presentation-ready insights — all in under 60 seconds.**

Built with Python · Powered by Claude · Designed for CPG marketing teams

---

## What This Does

Marketing Mix Modeling (MMM) tells you how much each advertising channel — TV, Digital, Radio, Print — contributed to your sales and what return you earned per dollar spent.

This POC wraps that analysis in a **natural-language AI interface** so marketing leaders can ask questions like:

```
"What is the ROI of TV vs Digital?"
"Which channels are saturated?"
"What happens if we cut Facebook by 20%?"
"What drove sales in Q4?"
```

...and receive instant, quantified, executive-ready answers alongside publication-quality charts.

---

## Architecture — Agent-Based Design

The system is structured as a pipeline of specialised agents, each with a single clear responsibility.

```
User Prompt (natural language)
       │
       ▼
┌─────────────────┐
│   NLP Router    │  Classifies intent · Extracts entities (channel, %)
│  (Claude API /  │  Routes to the correct analysis workflow
│  Rule-based)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Pipeline                            │
│                                                              │
│  1. DataIngestionAgent      Load CSV or generate synthetic   │
│  2. DataValidationAgent     Schema · nulls · outliers        │
│  3. FeatureEngineeringAgent Adstock + Hill saturation        │
│  4. ModelingAgent           OLS regression · ROI · curves    │
│  5. InsightGenerationAgent  Business-language narratives     │
│  6. VisualizationAgent      6 presentation-ready charts      │
│  7. ResponseFormattingAgent Rich terminal output             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Charts + Executive Summary
```

### Key Modelling Concepts

| Concept | What it Does |
|---|---|
| **Adstock** | Geometric decay: past advertising continues to influence future sales |
| **Saturation** | Hill function: captures diminishing returns at high spend |
| **OLS Regression** | Estimates channel effectiveness coefficients with confidence intervals |
| **ROI** | Incremental sales contribution ÷ total channel spend |
| **Response Curves** | Visualise where each channel sits on its saturation curve |

---

## Project Structure

```
Claude_MMM_POC/
├── agents/
│   ├── data_ingestion_agent.py       # Load CSV or synthetic data
│   ├── data_validation_agent.py      # Quality checks & reporting
│   ├── feature_engineering_agent.py  # Adstock + saturation transforms
│   ├── modeling_agent.py             # OLS MMM · ROI · response curves
│   ├── insight_generation_agent.py   # Business narrative generation
│   ├── visualization_agent.py        # 6 executive charts
│   └── response_formatting_agent.py  # Rich terminal rendering
├── config/
│   └── settings.py                   # All tunable parameters
├── data/
│   └── sample_data.py                # Synthetic 4-year MMM dataset
├── interface/
│   └── nlp_router.py                 # Intent classification & routing
├── outputs/                          # Generated charts (gitignored)
├── tests/
│   └── test_agents.py                # Pytest unit tests (36 tests)
├── main.py                           # Interactive CLI entry point
├── demo.py                           # One-shot executive demo
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/Claude_MMM_POC.git
cd Claude_MMM_POC

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set up your Anthropic API key for advanced NLP
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

---

## Usage

### Option 1: Interactive Mode

```bash
python main.py
```

Ask questions in plain English at the prompt:

```
You: Show me channel ROI
You: Which channels are saturated?
You: What if we cut TV by 20%?
You: What drove Q4 performance?
You: demo
You: quit
```

### Option 2: Full Executive Demo (no interaction required)

```bash
python demo.py
```

Runs the complete 4-stage analysis, prints executive insights, and saves all 6 charts to `outputs/`.

### Option 3: Use Your Own Data

```bash
python main.py --csv path/to/your_data.csv
```

Your CSV must contain these columns:

| Column | Description |
|---|---|
| `date` | Week start date (any parseable format) |
| `sales` | Weekly sales ($000s) |
| `tv_spend` | TV media spend ($000s) |
| `radio_spend` | Radio media spend ($000s) |
| `digital_spend` | Digital media spend ($000s) |
| `print_spend` | Print media spend ($000s) |

### Option 4: Text-Only (No Charts)

```bash
python main.py --no-charts
```

---

## Example Prompts & Outputs

| User Question | Intent Detected | Output |
|---|---|---|
| `"What is the ROI of TV vs Digital?"` | `roi_analysis` | ROI table + bar chart |
| `"Which channels are saturated?"` | `saturation` | Saturation % per channel + S-curves |
| `"What's driving our sales?"` | `contribution` | % breakdown + stacked area chart |
| `"What if we cut TV by 20%?"` | `budget_scenario` | Estimated sales & ROI impact |
| `"Why were Q4 sales high?"` | `trend` | Trend analysis + quarterly rollup |
| `"Give me an executive summary"` | `executive_summary` | Full briefing + all 6 charts |

---

## Charts Generated

| # | Chart | Description |
|---|---|---|
| 1 | `01_sales_spend_trend.png` | Sales line + stacked media spend bars over time |
| 2 | `02_channel_contributions.png` | Stacked area: weekly contribution by channel |
| 3 | `03_roi_comparison.png` | Horizontal bar chart, ranked by ROI |
| 4 | `04_response_curves.png` | S-curves (saturation) for each channel |
| 5 | `05_budget_efficiency.png` | Spend vs contribution scatter / efficiency matrix |
| 6 | `06_decomposition_pie.png` | Sales decomposition donut chart |

---

## Running Tests

```bash
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=agents --cov=interface --cov-report=term-missing
```

36 unit tests covering all agents, intent classification, entity extraction, and model validity.

---

## Configuration

All tunable parameters live in [`config/settings.py`](config/settings.py):

```python
# Adstock decay rates (0 = no carryover, 1 = infinite carryover)
ADSTOCK_DECAY_RATES = {
    "TV":      0.70,   # Brand-building, long memory
    "Digital": 0.20,   # Short-lived, click-driven
    ...
}

# Hill saturation parameters
SATURATION_PARAMS = {
    "TV": {"alpha": 2.0, "K": 0.50},
    ...
}
```

Changing these values changes how the model represents each channel's response characteristics.

---

## Extending the System

### Add a New Channel

1. Add the channel name to `MEDIA_CHANNELS` in `config/settings.py`
2. Add its adstock and saturation parameters
3. Ensure your dataset CSV includes a `{channel_lower}_spend` column

### Add a New Analysis Intent

1. Add the intent + keywords to `INTENTS` in `interface/nlp_router.py`
2. Add a method to `InsightGenerationAgent`
3. Add routing logic in the `dispatch()` function in `main.py`

### Connect Real Data

Replace the synthetic data generator with your data warehouse connection in `DataIngestionAgent.load_csv()` — the rest of the pipeline is data-source agnostic.

---

## Future Enhancements

| Enhancement | Value |
|---|---|
| Bayesian MMM (PyMC / LightweightMMM) | Uncertainty quantification on ROI estimates |
| Budget optimiser (scipy.optimize) | Find the spend allocation that maximises predicted sales |
| Streamlit / Gradio web UI | Browser-based dashboard for non-technical users |
| Live data connectors | Google Ads, Meta, GA4 API ingestion |
| Multi-market / geo-level MMM | Country or region-level attribution |
| Time-varying parameters | Capture structural breaks (COVID, new competitors) |
| Automated report export | PDF/PowerPoint slide generation |

---

## Business Context

This POC was designed to answer the questions a **CMO or VP Marketing** asks every quarter:

- *"Which channels actually drive incremental sales?"*
- *"Are we over-spending on TV?"*
- *"If we shift 20% of our TV budget to Digital, what happens?"*
- *"How efficient is our marketing mix vs industry benchmarks?"*

The system is designed to be **transparent** (explainable OLS model), **fast** (seconds, not hours), and **executive-ready** (plain English + charts).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data & modelling | pandas, numpy, statsmodels |
| Visualisation | matplotlib, seaborn |
| Terminal UI | rich |
| NLP routing | Rule-based + Claude API (optional) |
| Testing | pytest |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset structure inspired by [Google LightweightMMM](https://github.com/google/lightweight_mmm)
- Built with [Claude](https://claude.ai) by Anthropic

---

*Built as a proof of concept for AI-augmented marketing analytics. Not intended for production use without additional validation.*
