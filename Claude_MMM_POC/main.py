"""
main.py -- MMM AI System: Interactive CLI Entry Point

Usage:
    python main.py                      # Interactive mode
    python main.py --demo               # Run full executive demo
    python main.py --csv path/to/data.csv  # Use your own dataset

Architecture:
    DataIngestionAgent  ->  DataValidationAgent  ->  FeatureEngineeringAgent
    ->  ModelingAgent  ->  InsightGenerationAgent  +  VisualizationAgent
    ->  ResponseFormattingAgent  <-  NLPRouter  <-  User prompt
"""

from __future__ import annotations

import sys
import argparse
from typing import Optional, Dict, Any
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so rich can write Unicode panel borders
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# == Optional rich imports ===================================================
try:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False

from agents import (
    DataIngestionAgent,
    DataValidationAgent,
    FeatureEngineeringAgent,
    ModelingAgent,
    InsightGenerationAgent,
    VisualizationAgent,
    ResponseFormattingAgent,
    BudgetOptimizationAgent,
)
from interface.nlp_router import NLPRouter


# =============================================================================
# System Initialisation
# =============================================================================

def build_system(csv_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialise all agents and fit the MMM.

    Returns a dict with the fitted model and all agents ready to use.
    """
    console = Console(force_terminal=True, legacy_windows=False) if _RICH else None

    def _print(msg: str) -> None:
        if _RICH:
            console.print(msg)
        else:
            # Strip rich markup for plain mode
            import re
            print(re.sub(r"\[.*?\]", "", msg))

    _print("\n[bold cyan]Initialising Marketing Mix Model System...[/bold cyan]\n")

    # Step 1: Ingest
    _print("  [dim]1/5[/dim] Loading data...")
    ingestion = DataIngestionAgent()
    if csv_path:
        df_raw = ingestion.load_csv(csv_path)
    else:
        df_raw = ingestion.load_synthetic()
    _print(f"       [green]OK[/green] {ingestion.source} -- {len(df_raw)} weeks loaded")

    # Step 2: Validate
    _print("  [dim]2/5[/dim] Validating dataset...")
    validator = DataValidationAgent()
    report    = validator.validate(df_raw)
    if not report["is_valid"]:
        _print("\n[bold red]Data validation FAILED:[/bold red]")
        for err in report["errors"]:
            _print(f"  [red]x[/red] {err}")
        sys.exit(1)
    warn_count = len(report["warnings"])
    _print(f"       [green]OK[/green] Valid  ({warn_count} warning(s))")
    if warn_count and _RICH:
        for w in report["warnings"]:
            console.print(f"         [yellow]!![/yellow] {w}")

    # Step 3: Feature engineering
    _print("  [dim]3/5[/dim] Engineering features (adstock + saturation)...")
    fe_agent   = FeatureEngineeringAgent()
    df_eng     = fe_agent.engineer_all_features(df_raw)
    _print("       [green]OK[/green] Adstock & Hill-function saturation applied")

    # Step 4: Fit model
    _print("  [dim]4/5[/dim] Fitting OLS Marketing Mix Model...")
    model_agent = ModelingAgent()
    model_agent.fit(df_eng)
    fit = model_agent.get_fit_quality()
    _print(f"       [green]OK[/green] R² = {fit['R²']}  |  MAPE = {fit['MAPE (%)']:.1f}%")

    # Step 5: Prepare insight & viz agents
    _print("  [dim]5/5[/dim] Preparing insight and visualization engines...")
    insight_agent = InsightGenerationAgent(model_agent)
    viz_agent     = VisualizationAgent(model_agent)
    formatter     = ResponseFormattingAgent()
    router        = NLPRouter(use_claude_api=True)
    _print("       [green]OK[/green] System ready\n")

    return {
        "df":      df_eng,
        "model":   model_agent,
        "insight": insight_agent,
        "viz":     viz_agent,
        "format":  formatter,
        "router":  router,
    }


# =============================================================================
# Question -> Analysis Dispatcher
# =============================================================================

def dispatch(
    question: str,
    system: Dict[str, Any],
    generate_charts: bool = True,
) -> None:
    """
    Route a user question through the NLP router and run the analysis.

    Args:
        question:        User's natural-language input.
        system:          Initialised system dict from build_system().
        generate_charts: Whether to save chart images.
    """
    router  = system["router"]
    insight = system["insight"]
    viz     = system["viz"]
    fmt     = system["format"]

    intent, entities = router.parse(question)

    # Check for missing entities before proceeding
    clarify = router.get_clarifying_question(intent, entities)
    if clarify:
        if _RICH:
            Console(force_terminal=True, legacy_windows=False).print(f"\n[yellow]?[/yellow] {clarify}\n")
        else:
            print(f"\n? {clarify}\n")
        return

    chart_paths: Dict = {}

    # == Route to analysis ==================================================
    if intent == "roi_analysis":
        result = insight.roi_analysis()
        if generate_charts:
            chart_paths = {"roi_comparison": viz.plot_roi_comparison()}

    elif intent == "saturation":
        result = insight.saturation_analysis()
        if generate_charts:
            chart_paths = {"response_curves": viz.plot_response_curves()}

    elif intent == "contribution":
        result = insight.contribution_analysis()
        if generate_charts:
            chart_paths = {
                "channel_contributions": viz.plot_channel_contributions(),
                "decomposition_pie":     viz.plot_decomposition_pie(),
            }

    elif intent == "budget_scenario":
        channel = entities.get("channel", "Digital")
        pct     = entities.get("pct", -0.20)
        result  = insight.budget_scenario(channel, pct)
        chart_paths = {}   # No chart for scenarios (text-only)

    elif intent == "trend":
        period = entities.get("quarter")
        result = insight.trend_analysis(period=period)
        if generate_charts:
            chart_paths = {"sales_spend_trend": viz.plot_sales_spend_trend()}

    elif intent == "budget_optimize":
        total_budget = entities.get("total_budget")
        if total_budget is None:
            # Default: use sum of current channel spends
            total_budget = sum(
                system["model"].df[col].mean()
                for col in ["tv_spend", "radio_spend", "digital_spend", "print_spend"]
            )
        opt_agent = BudgetOptimizationAgent(system["model"])
        result = opt_agent.optimize(total_budget=total_budget)
        chart_paths = {}

    elif intent in ("executive_summary", "unknown"):
        result = insight.executive_summary()
        if generate_charts:
            chart_paths = viz.generate_all_charts()

    elif intent == "help":
        fmt.render_help()
        return

    elif intent == "demo":
        run_demo(system)
        return

    else:
        result = insight.executive_summary()
        if generate_charts:
            chart_paths = viz.generate_all_charts()

    fmt.render(result, chart_paths if generate_charts else None)


def run_demo(system: Dict[str, Any]) -> None:
    """Run the full executive demo -- generates all charts and prints summary."""
    fmt = system["format"]

    if _RICH:
        Console(force_terminal=True, legacy_windows=False).print(
            "\n[bold magenta]Running Full Executive Demo...[/bold magenta]\n"
        )
    else:
        print("\nRunning Full Executive Demo...\n")

    result      = system["insight"].executive_summary()
    chart_paths = system["viz"].generate_all_charts()
    fmt.render(result, chart_paths)

    if _RICH:
        Console(force_terminal=True, legacy_windows=False).print(
            f"\n[bold green]Demo complete![/bold green] "
            f"{len(chart_paths)} charts saved to: outputs/\n"
        )
    else:
        print(f"\nDemo complete! {len(chart_paths)} charts saved to: outputs/\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marketing Mix Model AI -- Executive Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive mode
  python main.py --demo                   # Full executive demo
  python main.py --csv data/my_data.csv   # Use your own CSV
        """,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run the full executive demo analysis")
    parser.add_argument("--csv",  type=str, default=None,
                        help="Path to a custom CSV dataset")
    parser.add_argument("--no-charts", action="store_true",
                        help="Disable chart generation (text-only output)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    system = build_system(csv_path=args.csv)
    fmt    = system["format"]

    if args.demo:
        run_demo(system)
        return

    # == Interactive Loop ===================================================
    fmt.render_welcome()
    fmt.render_help()

    generate_charts = not args.no_charts

    while True:
        try:
            if _RICH:
                question = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            else:
                question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q", "bye"):
                if _RICH:
                    Console(force_terminal=True, legacy_windows=False).print("[bold]Goodbye![/bold]")
                else:
                    print("Goodbye!")
                break

            dispatch(question, system, generate_charts=generate_charts)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
