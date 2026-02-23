"""
visualization_agent.py -- Agent 6: Visualization

Responsibilities:
  * Generate presentation-ready charts for all MMM analyses
  * Save charts to the outputs/ directory
  * Return file paths for downstream use

Charts produced:
  1. sales_spend_trend      -- Sales & media spend over time
  2. channel_contributions  -- Stacked area contribution decomposition
  3. roi_comparison         -- ROI bar chart (ranked)
  4. response_curves        -- Saturation S-curves per channel
  5. budget_efficiency      -- Spend vs contribution scatter
  6. decomposition_pie      -- Sales decomposition donut chart
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works headless / on Windows)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import (
    MEDIA_CHANNELS,
    SPEND_COLS,
    CHANNEL_COLORS,
    FIGURE_SIZE,
    FIGURE_SIZE_WIDE,
    FIGURE_SIZE_SQUARE,
    DPI,
    CHART_STYLE,
    OUTPUTS_DIR,
    COMPANY_NAME,
    CURRENCY_SYMBOL,
)


class VisualizationAgent:
    """
    Creates, saves, and catalogues all MMM charts.

    Usage:
        viz = VisualizationAgent(modeling_agent)
        paths = viz.generate_all_charts()
    """

    def __init__(self, modeling_agent) -> None:
        self.model = modeling_agent
        self.saved_paths: List[Path] = []
        try:
            plt.style.use(CHART_STYLE)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")

    # == Public Interface ====================================================

    def generate_all_charts(self) -> Dict[str, Path]:
        """
        Generate and save all six standard MMM charts.

        Returns:
            Dict mapping chart name -> saved file path.
        """
        results = {}
        chart_funcs = {
            "sales_spend_trend":     self.plot_sales_spend_trend,
            "channel_contributions": self.plot_channel_contributions,
            "roi_comparison":        self.plot_roi_comparison,
            "response_curves":       self.plot_response_curves,
            "budget_efficiency":     self.plot_budget_efficiency,
            "decomposition_pie":     self.plot_decomposition_pie,
        }
        for name, func in chart_funcs.items():
            try:
                path = func()
                results[name] = path
            except Exception as e:
                print(f"  ⚠ Could not generate '{name}': {e}")

        self.saved_paths = list(results.values())
        return results

    # == Individual Chart Methods ============================================

    def plot_sales_spend_trend(self) -> Path:
        """
        Dual-axis time-series: sales (line) + stacked media spend (bars).
        """
        df = self.model.df
        fig, ax1 = plt.subplots(figsize=FIGURE_SIZE_WIDE)

        # Stacked spend bars
        bottom = np.zeros(len(df))
        for ch in MEDIA_CHANNELS:
            col   = SPEND_COLS[ch]
            vals  = df[col].to_numpy()
            ax1.bar(df["date"], vals, bottom=bottom,
                    color=CHANNEL_COLORS[ch], alpha=0.75,
                    label=f"{ch} Spend", width=6)
            bottom += vals

        ax1.set_xlabel("Week", fontsize=11)
        ax1.set_ylabel(f"Media Spend ({CURRENCY_SYMBOL}000s)", fontsize=11)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))

        # Sales line (secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(df["date"], df["sales"], color="#212121", linewidth=2.0,
                 label="Actual Sales", zorder=5)
        ax2.set_ylabel(f"Sales ({CURRENCY_SYMBOL}000s)", fontsize=11)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))

        # Legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2,
                   loc="upper left", fontsize=9, framealpha=0.9)

        self._style(fig, ax1,
                    title=f"{COMPANY_NAME} -- Sales & Media Spend Over Time",
                    subtitle="Weekly actuals | Stacked bars = media spend by channel")
        return self._save(fig, "01_sales_spend_trend.png")

    def plot_channel_contributions(self) -> Path:
        """
        Stacked area chart showing weekly sales contribution by component.
        """
        dec  = self.model.decomposition
        df   = self.model.df
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

        components = ["Base"] + MEDIA_CHANNELS
        colors     = [CHANNEL_COLORS.get(c, "#999999") for c in components]

        bottom = np.zeros(len(dec))
        for comp, color in zip(components, colors):
            vals = np.maximum(dec[comp].to_numpy(), 0)
            ax.fill_between(df["date"], bottom, bottom + vals,
                            alpha=0.80, color=color, label=comp)
            bottom += vals

        ax.plot(df["date"], dec["actual"], color="#212121",
                linewidth=1.5, linestyle="--", label="Actual Sales", zorder=10)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
        ax.set_xlabel("Week", fontsize=11)
        ax.set_ylabel(f"Sales ({CURRENCY_SYMBOL}000s)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

        self._style(fig, ax,
                    title=f"{COMPANY_NAME} -- Sales Contribution Decomposition",
                    subtitle="Stacked area = modelled contribution by channel | Dashed = actual sales")
        return self._save(fig, "02_channel_contributions.png")

    def plot_roi_comparison(self) -> Path:
        """
        Horizontal bar chart of channel ROI, ranked best -> worst.
        """
        roi     = self.model.roi
        ranked  = sorted(roi.items(), key=lambda x: x[1])   # ascending for horizontal bar
        channels = [r[0] for r in ranked]
        values   = [r[1] for r in ranked]
        colors   = [CHANNEL_COLORS.get(c, "#999") for c in channels]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(channels, values, color=colors, edgecolor="white",
                       linewidth=0.8, height=0.55)

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"  {val:.2f}x", va="center", fontsize=11, fontweight="bold")

        ax.set_xlabel(f"Return on Investment ({CURRENCY_SYMBOL} per {CURRENCY_SYMBOL}1 spent)", fontsize=11)
        ax.set_title(f"{COMPANY_NAME} -- Channel ROI Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.axvline(x=1.0, color="#E53935", linestyle="--", linewidth=1.2,
                   label="Break-even (1.0x)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, max(values) * 1.2)

        plt.tight_layout()
        fig.text(0.5, -0.02,
                 "Note: ROI = incremental sales contribution / total channel spend",
                 ha="center", fontsize=8, color="#666666")
        return self._save(fig, "03_roi_comparison.png")

    def plot_response_curves(self) -> Path:
        """
        2x2 grid of saturation S-curves -- one per channel.
        Shows current spend position on each curve.
        """
        curves = self.model.get_response_curves()
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes_flat = axes.flatten()

        for i, ch in enumerate(MEDIA_CHANNELS):
            ax      = axes_flat[i]
            curve   = curves[ch]
            color   = CHANNEL_COLORS.get(ch, "#333")

            ax.plot(curve["spend"], curve["sales_lift"],
                    color=color, linewidth=2.5, label="Response curve")
            ax.axvline(x=curve["current_spend"], color="#E53935",
                       linestyle="--", linewidth=1.5, label="Current spend")
            ax.scatter([curve["current_spend"]], [curve["current_lift"]],
                       color="#E53935", s=80, zorder=5)
            ax.fill_between(curve["spend"], curve["sales_lift"],
                            alpha=0.12, color=color)

            max_lift = curve["sales_lift"].max()
            pct_sat  = curve["current_lift"] / max_lift * 100 if max_lift > 0 else 0

            ax.set_title(f"{ch}  *  {pct_sat:.0f}% saturated",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel(f"Weekly Spend ({CURRENCY_SYMBOL}000s)", fontsize=9)
            ax.set_ylabel("Sales Lift ($000s)", fontsize=9)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
            ax.legend(fontsize=8)

        fig.suptitle(f"{COMPANY_NAME} -- Response / Saturation Curves",
                     fontsize=15, fontweight="bold", y=1.01)
        fig.text(0.5, -0.01,
                 "Red dashed line = current mean weekly spend | Curve shows modelled sales lift",
                 ha="center", fontsize=8, color="#555")
        plt.tight_layout()
        return self._save(fig, "04_response_curves.png")

    def plot_budget_efficiency(self) -> Path:
        """
        Scatter plot: total spend (x) vs sales contribution (y).
        Bubble size proportional to ROI.
        """
        df   = self.model.df
        roi  = self.model.roi
        contribs = self.model.contributions

        spend_totals = {ch: df[SPEND_COLS[ch]].sum() for ch in MEDIA_CHANNELS}
        contrib_totals = {ch: contribs[ch].sum() for ch in MEDIA_CHANNELS}

        fig, ax = plt.subplots(figsize=(9, 7))

        for ch in MEDIA_CHANNELS:
            x     = spend_totals[ch]
            y     = contrib_totals[ch]
            r     = roi[ch]
            color = CHANNEL_COLORS.get(ch, "#999")
            size  = max(200, r * 600)

            ax.scatter(x, y, s=size, color=color, alpha=0.80,
                       edgecolors="white", linewidth=1.5, zorder=5)
            ax.annotate(f"  {ch}\n  ROI: {r:.2f}x",
                        (x, y), fontsize=10, fontweight="bold",
                        color=color, va="center")

        # Break-even line: contribution = spend
        xlim = ax.get_xlim()
        line_x = np.linspace(0, max(spend_totals.values()) * 1.2, 100)
        ax.plot(line_x, line_x, color="#E53935", linestyle="--",
                linewidth=1.2, label="Break-even (1:1)")

        ax.set_xlabel(f"Total Media Spend ({CURRENCY_SYMBOL}000s)", fontsize=11)
        ax.set_ylabel(f"Total Sales Contribution ({CURRENCY_SYMBOL}000s)", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:,.0f}M"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:,.0f}M"))
        ax.legend(fontsize=9)

        self._style(fig, ax,
                    title=f"{COMPANY_NAME} -- Budget Efficiency Matrix",
                    subtitle="Bubble size = ROI magnitude | Above break-even line = positive return")
        return self._save(fig, "05_budget_efficiency.png")

    def plot_decomposition_pie(self) -> Path:
        """
        Donut chart showing percentage sales contribution by component.
        """
        contribs    = self.model.contributions
        total_sales = self.model.df["sales"].sum()

        components = ["Base"] + MEDIA_CHANNELS
        values     = [max(0, contribs[c].sum()) for c in components]
        colors     = [CHANNEL_COLORS.get(c, "#999") for c in components]
        pcts       = [v / total_sales * 100 for v in values]
        labels     = [f"{c}\n{p:.1f}%" for c, p in zip(components, pcts)]

        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
        wedges, texts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 11},
        )

        ax.text(0, 0, f"${total_sales/1000:.1f}M\nTotal Sales",
                ha="center", va="center", fontsize=12, fontweight="bold")

        ax.set_title(f"{COMPANY_NAME} -- Sales Decomposition",
                     fontsize=14, fontweight="bold", pad=20)
        fig.text(0.5, 0.01,
                 "Inner label = % of total sales attributed to each component",
                 ha="center", fontsize=8, color="#555")
        plt.tight_layout()
        return self._save(fig, "06_decomposition_pie.png")

    # == Utilities ===========================================================

    @staticmethod
    def _style(fig: plt.Figure, ax: plt.Axes,
               title: str, subtitle: str = "") -> None:
        """Apply consistent title/subtitle styling."""
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        if subtitle:
            fig.text(0.5, 1.01, subtitle,
                     ha="center", fontsize=8, color="#555555",
                     transform=ax.transAxes)
        plt.tight_layout()

    @staticmethod
    def _save(fig: plt.Figure, filename: str) -> Path:
        """Save figure to outputs directory and close it."""
        path = OUTPUTS_DIR / filename
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return path
