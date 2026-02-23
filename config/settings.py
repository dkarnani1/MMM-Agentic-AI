"""
settings.py — Central configuration for the MMM AI POC application.

All agents import from here, keeping tuning parameters in one place.
"""

from pathlib import Path
from typing import Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Directory Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = BASE_DIR / "data"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

# Ensure output folder exists
OUTPUTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Media Channels
# ─────────────────────────────────────────────────────────────────────────────
MEDIA_CHANNELS: List[str] = ["TV", "Radio", "Digital", "Print"]

# Spend column names in the raw dataset
SPEND_COLS: Dict[str, str] = {ch: f"{ch.lower()}_spend" for ch in MEDIA_CHANNELS}

# ─────────────────────────────────────────────────────────────────────────────
# Adstock Parameters — Geometric Decay Rate per Channel
#   Higher value = longer carryover effect (TV persists longer than Digital)
# ─────────────────────────────────────────────────────────────────────────────
ADSTOCK_DECAY_RATES: Dict[str, float] = {
    "TV":      0.70,   # Brand-building; strong carryover
    "Radio":   0.40,   # Moderate recall
    "Digital": 0.20,   # Short-lived; click-driven
    "Print":   0.50,   # Magazine/newspaper; moderate
}

# ─────────────────────────────────────────────────────────────────────────────
# Saturation Parameters — Hill Function  f(x) = x^α / (x^α + K^α)
#   alpha : steepness of the S-curve
#   K     : half-saturation point (as fraction of normalised spend)
# ─────────────────────────────────────────────────────────────────────────────
SATURATION_PARAMS: Dict[str, Dict[str, float]] = {
    "TV":      {"alpha": 2.0, "K": 0.50},
    "Radio":   {"alpha": 1.5, "K": 0.40},
    "Digital": {"alpha": 2.5, "K": 0.30},
    "Print":   {"alpha": 1.8, "K": 0.45},
}

# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
CHANNEL_COLORS: Dict[str, str] = {
    "TV":          "#2196F3",   # Blue
    "Radio":       "#FF9800",   # Amber
    "Digital":     "#4CAF50",   # Green
    "Print":       "#9C27B0",   # Purple
    "Base":        "#607D8B",   # Blue-Grey
    "Seasonality": "#FF5722",   # Deep Orange
    "Trend":       "#795548",   # Brown
    "Total":       "#212121",   # Near-black
}

FIGURE_SIZE = (14, 7)
FIGURE_SIZE_WIDE = (16, 8)
FIGURE_SIZE_SQUARE = (10, 10)
DPI = 150
CHART_STYLE = "seaborn-v0_8-whitegrid"

# ─────────────────────────────────────────────────────────────────────────────
# Business / Branding
# ─────────────────────────────────────────────────────────────────────────────
COMPANY_NAME: str = "Company ABC"
REPORT_TITLE: str = "Marketing Mix Model — Executive Insights"
CURRENCY_SYMBOL: str = "$"
SALES_UNIT: str = "$000s"   # All monetary values are in thousands

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED: int = 42
N_WEEKS: int = 208          # 4 years of weekly data
RESPONSE_CURVE_POINTS: int = 100
