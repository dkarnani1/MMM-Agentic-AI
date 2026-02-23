"""
response_formatting_agent.py -- Agent 7: Response Formatting

Responsibilities:
  * Format insight dicts into rich terminal output using the `rich` library
  * Render tables, panels, and styled text for executive readability
  * Print chart save paths so the user knows where to find visuals
  * Produce plain-text fallback when `rich` is unavailable
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.text import Text
    from rich.columns import Columns
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class ResponseFormattingAgent:
    """
    Renders analysis results to the terminal in a professional, readable format.

    Usage:
        formatter = ResponseFormattingAgent()
        formatter.render(insight_dict, chart_paths)
    """

    def __init__(self, width: int = 100) -> None:
        if _RICH_AVAILABLE:
            # force_terminal=True bypasses legacy Windows console renderer
            self._console = Console(width=width, force_terminal=True, legacy_windows=False)
        else:
            self._console = None

    # == Public Interface ====================================================

    def render(
        self,
        insight: Dict[str, Any],
        chart_paths: Optional[Dict[str, Path]] = None,
    ) -> None:
        """
        Render an insight dict to the terminal.

        Args:
            insight:     Output from InsightGenerationAgent (any type).
            chart_paths: Optional dict of {chart_name: file_path}.
        """
        insight_type = insight.get("type", "unknown")

        if _RICH_AVAILABLE:
            self._render_rich(insight, chart_paths)
        else:
            self._render_plain(insight, chart_paths)

    def render_welcome(self) -> None:
        """Print the application welcome banner."""
        if _RICH_AVAILABLE:
            self._console.print(
                Panel.fit(
                    "[bold cyan]Marketing Mix Model AI System[/bold cyan]\n"
                    "[dim]Powered by Claude  |  Built for Executive Decision-Making[/dim]",
                    border_style="cyan",
                    padding=(1, 4),
                )
            )
        else:
            print("\n" + "=" * 60)
            print("  MARKETING MIX MODEL AI SYSTEM")
            print("  Powered by Claude  |  Built for Executive Decision-Making")
            print("=" * 60 + "\n")

    def render_system_status(self, steps: List[str], current: int) -> None:
        """Show initialisation progress steps."""
        if _RICH_AVAILABLE:
            for i, step in enumerate(steps):
                icon = "[green]OK[/green]" if i < current else "[dim]--[/dim]"
                self._console.print(f"  {icon} {step}")
        else:
            for i, step in enumerate(steps):
                icon = "OK" if i < current else "--"
                print(f"  {icon} {step}")

    def render_help(self) -> None:
        """Display example prompts and usage tips."""
        examples = [
            ("ROI Analysis",        "Show me channel ROI"),
            ("Saturation",          "Which channels are saturated?"),
            ("Contributions",       "What's driving sales?"),
            ("Budget Scenario",     "What if we cut TV by 20%?"),
            ("Trend",               "What drove Q4 performance?"),
            ("Overview",            "Give me an executive summary"),
            ("Demo Mode",           "demo"),
        ]

        if _RICH_AVAILABLE:
            table = Table(title="Example Prompts", box=box.SIMPLE_HEAD,
                          show_header=True, header_style="bold cyan")
            table.add_column("Intent",  style="bold", width=22)
            table.add_column("Example Prompt", style="green")
            for intent, prompt in examples:
                table.add_row(intent, f'"{prompt}"')
            self._console.print(table)
            self._console.print(
                "\n[dim]Tip: Type 'quit' or press Ctrl+C to exit.[/dim]\n"
            )
        else:
            print("\nExample Prompts:")
            for intent, prompt in examples:
                print(f"  {intent:<22} -> \"{prompt}\"")
            print()

    # == Private Renderers ===================================================

    def _render_rich(
        self,
        insight: Dict[str, Any],
        chart_paths: Optional[Dict[str, Path]],
    ) -> None:
        c = self._console

        # Narrative / headline
        narrative = insight.get("narrative") or insight.get("headline", "")
        if narrative:
            c.print(Panel(narrative, border_style="blue", padding=(0, 2)))

        # Table (if present)
        table_data: List[Dict] = insight.get("table_data", [])
        if table_data:
            t = Table(box=box.SIMPLE_HEAD, show_header=True,
                      header_style="bold white on #1565C0")
            for col in table_data[0].keys():
                t.add_column(col, style="white")
            for row in table_data:
                t.add_row(*[str(v) for v in row.values()])
            c.print(t)

        # Key insights
        key_insights: List[str] = insight.get("key_insights", [])
        if key_insights:
            c.print("\n[bold cyan]Key Insights[/bold cyan]")
            for item in key_insights:
                c.print(f"  {item}")

        # Risks
        risks: List[str] = insight.get("risks", [])
        if risks:
            c.print("\n[bold yellow]Risks / Caveats[/bold yellow]")
            for r in risks:
                c.print(f"  [yellow]!![/yellow] {r}")

        # Actions
        actions: List[str] = insight.get("actions") or insight.get("top_actions", [])
        if actions:
            c.print("\n[bold green]Recommended Actions[/bold green]")
            for a in actions:
                c.print(f"  [green]->[/green] {a}")

        # Chart paths
        if chart_paths:
            c.print("\n[bold]Charts saved:[/bold]")
            for name, path in chart_paths.items():
                c.print(f"  [dim]>> {name}:[/dim] {path}")

        c.print()

    def _render_plain(
        self,
        insight: Dict[str, Any],
        chart_paths: Optional[Dict[str, Path]],
    ) -> None:
        sep = "=" * 60

        narrative = insight.get("narrative") or insight.get("headline", "")
        if narrative:
            print(f"\n{sep}")
            print(narrative)
            print(sep)

        table_data: List[Dict] = insight.get("table_data", [])
        if table_data:
            cols = list(table_data[0].keys())
            widths = [max(len(c), max(len(str(r[c])) for r in table_data)) for c in cols]
            header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
            print(f"\n{header}")
            print("=" * len(header))
            for row in table_data:
                print("  ".join(str(row[c]).ljust(w) for c, w in zip(cols, widths)))

        for section, title in [
            ("key_insights", "KEY INSIGHTS"),
            ("risks",        "RISKS / CAVEATS"),
            ("actions",      "RECOMMENDED ACTIONS"),
            ("top_actions",  "RECOMMENDED ACTIONS"),
        ]:
            items = insight.get(section, [])
            if items:
                print(f"\n{title}")
                for item in items:
                    print(f"  {item}")

        if chart_paths:
            print("\nCharts saved:")
            for name, path in chart_paths.items():
                print(f"  {name}: {path}")
        print()
