"""
demo.py -- Executive Demo Mode

Automatically runs a complete Marketing Mix Model analysis and generates
a full executive briefing package without any user interaction.

Output:
  * 6 presentation-ready charts saved to outputs/
  * Executive summary printed to terminal
  * Validation report
  * Model fit statistics

Usage:
    python demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so rich can write Unicode panel borders
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# == Optional rich imports ===================================================
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    _RICH = True
    # force_terminal=True bypasses the legacy Windows console renderer
    # so Unicode chars (if any remain) don't cause cp1252 encoding errors
    console = Console(force_terminal=True, legacy_windows=False)
except ImportError:
    _RICH = False
    console = None


def _print(msg: str) -> None:
    if _RICH:
        console.print(msg)
    else:
        import re
        print(re.sub(r"\[.*?\]", "", msg))


def _header(text: str) -> None:
    if _RICH:
        console.print(Panel(f"[bold white]{text}[/bold white]",
                            border_style="cyan", padding=(0, 2)))
    else:
        print(f"\n{'='*60}\n  {text}\n{'='*60}")


def run_demo() -> None:
    start = time.time()

    # == Banner =============================================================
    if _RICH:
        console.print(
            Panel.fit(
                "[bold cyan]Marketing Mix Model AI -- Executive Demo[/bold cyan]\n"
                "[dim]Full automated analysis  |  All charts generated  |  Executive briefing[/dim]",
                border_style="cyan",
                padding=(1, 6),
            )
        )
    else:
        print("\n" + "="*60)
        print("  MARKETING MIX MODEL AI -- EXECUTIVE DEMO")
        print("="*60 + "\n")

    # == Import agents ======================================================
    _print("\n[bold]Loading agents...[/bold]")
    from agents import (
        DataIngestionAgent,
        DataValidationAgent,
        FeatureEngineeringAgent,
        ModelingAgent,
        InsightGenerationAgent,
        VisualizationAgent,
        ResponseFormattingAgent,
    )

    # ======================================================================
    # STAGE 1: DATA
    # ======================================================================
    _header("Stage 1 of 4 -- Data Ingestion & Validation")

    ingestion = DataIngestionAgent()
    df_raw    = ingestion.load_synthetic()
    summary   = ingestion.get_summary()

    _print(f"\n[green]OK[/green] Dataset: {summary['source']}")
    _print(f"[green]OK[/green] {summary['rows']} weeks  |  {summary['date_range']}")
    _print(f"[green]OK[/green] Total sales: {summary['total_sales']}")

    _print("\n[dim]Spend by channel:[/dim]")
    for ch, spend in summary["total_spend"].items():
        _print(f"  {ch:<10} {spend}")

    validator = DataValidationAgent()
    v_report  = validator.validate(df_raw)
    status    = "[green]PASS[/green]" if v_report["is_valid"] else "[red]FAIL[/red]"
    _print(f"\n[bold]Validation:[/bold] {status}")
    if v_report["warnings"]:
        for w in v_report["warnings"]:
            _print(f"  [yellow]!![/yellow] {w}")

    # ======================================================================
    # STAGE 2: FEATURE ENGINEERING + MODELLING
    # ======================================================================
    _header("Stage 2 of 4 -- Feature Engineering & Modelling")

    _print("\n[dim]Applying adstock transformations...[/dim]")
    fe_agent = FeatureEngineeringAgent()
    df_eng   = fe_agent.engineer_all_features(df_raw)
    _print("[green]OK[/green] Adstock (geometric decay) applied to all channels")
    _print("[green]OK[/green] Hill-function saturation applied (diminishing returns)")
    _print("[green]OK[/green] Fourier seasonality features engineered")

    _print("\n[dim]Fitting OLS regression model...[/dim]")
    model   = ModelingAgent()
    model.fit(df_eng)
    fit     = model.get_fit_quality()

    _print(f"\n[bold]Model Fit Quality:[/bold]")
    for metric, val in fit.items():
        _print(f"  {metric:<20} {val}")

    # ======================================================================
    # STAGE 3: INSIGHTS
    # ======================================================================
    _header("Stage 3 of 4 -- Generating Executive Insights")

    insight_agent = InsightGenerationAgent(model)
    formatter     = ResponseFormattingAgent()

    # ROI Analysis
    _print("\n[bold cyan]== Channel ROI ==[/bold cyan]")
    roi_r = insight_agent.roi_analysis()
    formatter.render(roi_r)

    # Saturation
    _print("[bold cyan]== Saturation Assessment ==[/bold cyan]")
    sat_r = insight_agent.saturation_analysis()
    formatter.render(sat_r)

    # Contribution
    _print("[bold cyan]== Sales Contribution Breakdown ==[/bold cyan]")
    con_r = insight_agent.contribution_analysis()
    formatter.render(con_r)

    # Budget Scenario Example
    _print("[bold cyan]== Budget Scenario: 20% cut in lowest-ROI channel ==[/bold cyan]")
    worst_ch = min(model.roi, key=model.roi.get)
    scenario = insight_agent.budget_scenario(worst_ch, -0.20)
    formatter.render(scenario)

    # ======================================================================
    # STAGE 4: VISUALIZATIONS
    # ======================================================================
    _header("Stage 4 of 4 -- Generating Presentation Charts")

    viz     = VisualizationAgent(model)
    charts  = viz.generate_all_charts()

    _print(f"\n[bold green]OK {len(charts)} charts saved:[/bold green]")
    chart_labels = {
        "sales_spend_trend":     "Sales & Media Spend Timeline",
        "channel_contributions": "Sales Contribution Decomposition",
        "roi_comparison":        "Channel ROI Comparison",
        "response_curves":       "Saturation / Response Curves",
        "budget_efficiency":     "Budget Efficiency Matrix",
        "decomposition_pie":     "Sales Decomposition Donut",
    }
    for key, path in charts.items():
        label = chart_labels.get(key, key)
        _print(f"  >>{label}")
        _print(f"     [dim]{path}[/dim]")

    # ======================================================================
    # EXECUTIVE SUMMARY
    # ======================================================================
    _header("Executive Summary")
    exec_r = insight_agent.executive_summary()
    formatter.render(exec_r, charts)

    elapsed = time.time() - start
    _print(
        f"\n[bold green]Demo complete in {elapsed:.1f}s.[/bold green]  "
        f"Charts saved to: [cyan]outputs/[/cyan]\n"
    )


if __name__ == "__main__":
    run_demo()
