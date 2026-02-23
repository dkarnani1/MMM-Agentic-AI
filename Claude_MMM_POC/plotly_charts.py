"""
plotly_charts.py — Interactive Plotly chart builders for the MMM dashboard.

Each public method accepts a chart_type string and returns either:
  - plotly.graph_objects.Figure  (for all chart views)
  - pd.DataFrame                 (for the "table" view)

Supported chart types per method
  roi_comparison       : hbar | vbar | radar | table
  response_curves      : line | scatter | table
  channel_contributions: stacked_area | stacked_bar | line | table
  sales_spend_trend    : combo | line | bar | table
  budget_efficiency    : bubble | bar | table
  decomposition_pie    : donut | pie | bar | table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Union

from config.settings import (
    MEDIA_CHANNELS,
    SPEND_COLS,
    CHANNEL_COLORS,
    COMPANY_NAME,
    CURRENCY_SYMBOL,
)

FigOrDf = Union[go.Figure, pd.DataFrame]

# ── Shared layout defaults ────────────────────────────────────────────────────
_LAYOUT: dict = dict(
    template="plotly_white",
    font=dict(family="Arial, sans-serif", size=12),
    margin=dict(t=60, b=40, l=60, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
    # hovermode is set per chart to avoid duplicate-key errors when merging
)


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert #RRGGBB hex string to (R, G, B) integer tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Return CSS rgba() string from a hex colour."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ─────────────────────────────────────────────────────────────────────────────


class PlotlyCharts:
    """
    Interactive Plotly chart builders backed by a fitted ModelingAgent.

    Usage:
        pc = PlotlyCharts(modeling_agent)
        fig = pc.roi_comparison("hbar")      # go.Figure
        df  = pc.roi_comparison("table")     # pd.DataFrame
    """

    def __init__(self, modeling_agent) -> None:
        self.model = modeling_agent

    # ── ROI Comparison ───────────────────────────────────────────────────────

    def roi_comparison(self, chart_type: str = "hbar") -> FigOrDf:
        """Channel ROI bar/radar chart.  Types: hbar | vbar | radar | table."""
        roi = self.model.roi
        ranked_asc  = sorted(roi.items(), key=lambda x: x[1])
        ranked_desc = sorted(roi.items(), key=lambda x: -x[1])

        if chart_type == "table":
            return pd.DataFrame([
                {
                    "Channel":        ch,
                    "ROI":            f"{v:.2f}x",
                    "vs Break-even":  "Above 1.0x" if v >= 1 else "Below 1.0x",
                }
                for ch, v in ranked_desc
            ])

        if chart_type == "hbar":
            channels = [r[0] for r in ranked_asc]
            values   = [r[1] for r in ranked_asc]
            colors   = [CHANNEL_COLORS.get(c, "#999") for c in channels]
            fig = go.Figure(go.Bar(
                y=channels, x=values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}x" for v in values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.2f}x<extra></extra>",
            ))
            fig.add_vline(x=1.0, line_dash="dash", line_color="#E53935",
                          annotation_text="Break-even (1.0x)")
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Channel ROI Comparison",
                xaxis_title=f"ROI ({CURRENCY_SYMBOL} returned per {CURRENCY_SYMBOL}1 spent)",
            )

        elif chart_type == "vbar":
            channels = [r[0] for r in ranked_desc]
            values   = [r[1] for r in ranked_desc]
            colors   = [CHANNEL_COLORS.get(c, "#999") for c in channels]
            fig = go.Figure(go.Bar(
                x=channels, y=values,
                marker_color=colors,
                text=[f"{v:.2f}x" for v in values],
                textposition="outside",
                hovertemplate="%{x}: %{y:.2f}x<extra></extra>",
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="#E53935",
                          annotation_text="Break-even (1.0x)")
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Channel ROI Comparison",
                yaxis_title=f"ROI ({CURRENCY_SYMBOL} per {CURRENCY_SYMBOL}1 spent)",
            )

        elif chart_type == "radar":
            channels = [r[0] for r in ranked_asc]
            values   = [r[1] for r in ranked_asc]
            theta = channels + [channels[0]]
            r_vals = values + [values[0]]
            fig = go.Figure(go.Scatterpolar(
                r=r_vals,
                theta=theta,
                fill="toself",
                line=dict(color="#4CAF50", width=2),
                marker=dict(size=8),
                hovertemplate="%{theta}: %{r:.2f}x<extra></extra>",
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Channel ROI Radar",
                polar=dict(radialaxis=dict(visible=True,
                                           range=[0, max(values) * 1.25])),
            )

        return fig

    # ── Response / Saturation Curves ─────────────────────────────────────────

    def response_curves(self, chart_type: str = "line") -> FigOrDf:
        """Saturation S-curves.  Types: line | scatter | table."""
        curves = self.model.get_response_curves()

        if chart_type == "table":
            rows = []
            for ch in MEDIA_CHANNELS:
                c        = curves[ch]
                max_lift = float(c["sales_lift"].max())
                sat_pct  = c["current_lift"] / max_lift * 100 if max_lift > 0 else 0
                rows.append({
                    "Channel":                ch,
                    "Current Spend ($k)":     f"${c['current_spend']:,.0f}k",
                    "Current Lift ($k)":      f"${c['current_lift']:,.0f}k",
                    "Saturation %":           f"{sat_pct:.0f}%",
                    "Max Possible Lift ($k)": f"${max_lift:,.0f}k",
                    "Headroom ($k)":          f"${max_lift - c['current_lift']:,.0f}k",
                })
            return pd.DataFrame(rows)

        # Compute per-channel saturation % for subplot titles
        subplot_titles = []
        for ch in MEDIA_CHANNELS:
            c        = curves[ch]
            max_lift = float(c["sales_lift"].max())
            sat_pct  = c["current_lift"] / max_lift * 100 if max_lift > 0 else 0
            subplot_titles.append(f"{ch} — {sat_pct:.0f}% saturated")

        if chart_type == "line":
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=subplot_titles,
                vertical_spacing=0.18,
                horizontal_spacing=0.1,
            )
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for (row, col), ch in zip(positions, MEDIA_CHANNELS):
                c     = curves[ch]
                color = CHANNEL_COLORS.get(ch, "#333")
                max_y = float(c["sales_lift"].max())

                # Response curve fill
                fig.add_trace(go.Scatter(
                    x=c["spend"], y=c["sales_lift"],
                    mode="lines",
                    name=ch,
                    line=dict(color=color, width=2.5),
                    fill="tozeroy",
                    fillcolor=_rgba(color, 0.12),
                    showlegend=False,
                    hovertemplate=(
                        f"{ch}<br>Spend: $%{{x:,.0f}}k"
                        f"<br>Lift: $%{{y:,.0f}}k<extra></extra>"
                    ),
                ), row=row, col=col)

                # Vertical line at current spend (as a scatter trace)
                fig.add_trace(go.Scatter(
                    x=[c["current_spend"], c["current_spend"]],
                    y=[0, max_y],
                    mode="lines",
                    line=dict(color="#E53935", dash="dash", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ), row=row, col=col)

                # Marker at current position
                fig.add_trace(go.Scatter(
                    x=[c["current_spend"]],
                    y=[c["current_lift"]],
                    mode="markers",
                    marker=dict(color="#E53935", size=10),
                    showlegend=False,
                    hovertemplate=(
                        f"Current<br>${c['current_spend']:,.0f}k"
                        f" → ${c['current_lift']:,.0f}k lift<extra></extra>"
                    ),
                ), row=row, col=col)

            fig.update_layout(
                **{**_LAYOUT, "height": 580},
                title=f"{COMPANY_NAME} — Response / Saturation Curves",
            )

        elif chart_type == "scatter":
            fig = go.Figure()
            for ch in MEDIA_CHANNELS:
                c     = curves[ch]
                color = CHANNEL_COLORS.get(ch, "#333")
                step  = max(1, len(c["spend"]) // 50)
                fig.add_trace(go.Scatter(
                    x=c["spend"][::step],
                    y=c["sales_lift"][::step],
                    mode="markers+lines",
                    name=ch,
                    marker=dict(color=color, size=6),
                    line=dict(color=color, width=1.5),
                    hovertemplate=(
                        f"{ch}<br>Spend: $%{{x:,.0f}}k"
                        f"<br>Lift: $%{{y:,.0f}}k<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter(
                    x=[c["current_spend"]],
                    y=[c["current_lift"]],
                    mode="markers",
                    marker=dict(color=color, size=14, symbol="star"),
                    name=f"{ch} (current)",
                    showlegend=False,
                    hovertemplate=(
                        f"{ch} current<br>"
                        f"${c['current_spend']:,.0f}k → ${c['current_lift']:,.0f}k<extra></extra>"
                    ),
                ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Response Curves (All Channels Overlay)",
                xaxis_title="Weekly Spend ($k)",
                yaxis_title="Sales Lift ($k)",
            )

        return fig

    # ── Channel Contributions ────────────────────────────────────────────────

    def channel_contributions(self, chart_type: str = "stacked_area") -> FigOrDf:
        """Contribution decomposition.  Types: stacked_area | stacked_bar | line | table."""
        dec          = self.model.decomposition
        df           = self.model.df
        components   = ["Base"] + MEDIA_CHANNELS
        total_actual = float(dec["actual"].sum())
        dates        = df["date"].tolist()

        if chart_type == "table":
            return pd.DataFrame([
                {
                    "Component":                comp,
                    "Total Contribution ($k)":  f"${float(dec[comp].sum()):,.0f}k",
                    "% of Sales":               f"{float(dec[comp].sum()) / total_actual * 100:.1f}%",
                    "Weekly Average ($k)":       f"${float(dec[comp].mean()):,.0f}k",
                }
                for comp in components
            ])

        if chart_type == "stacked_area":
            fig = go.Figure()
            for comp in components:
                color = CHANNEL_COLORS.get(comp, "#999")
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=np.maximum(dec[comp].to_numpy(), 0),
                    name=comp,
                    stackgroup="one",
                    line=dict(color=color, width=0.5),
                    fillcolor=color,
                    mode="lines",
                    hovertemplate=f"{comp}: $%{{y:,.0f}}k<extra></extra>",
                ))
            fig.add_trace(go.Scatter(
                x=dates, y=dec["actual"],
                name="Actual Sales",
                line=dict(color="#212121", width=2, dash="dot"),
                mode="lines",
                hovertemplate="Actual: $%{y:,.0f}k<extra></extra>",
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Sales Contribution Decomposition",
                yaxis_title="Sales ($k)",
                hovermode="x unified",
            )

        elif chart_type == "stacked_bar":
            fig = go.Figure()
            for comp in components:
                color = CHANNEL_COLORS.get(comp, "#999")
                fig.add_trace(go.Bar(
                    x=dates,
                    y=np.maximum(dec[comp].to_numpy(), 0),
                    name=comp,
                    marker_color=color,
                    hovertemplate=f"{comp}: $%{{y:,.0f}}k<extra></extra>",
                ))
            fig.update_layout(
                **{**_LAYOUT, "height": 450},
                barmode="stack",
                title=f"{COMPANY_NAME} — Sales Contribution (Stacked Bars)",
                yaxis_title="Sales ($k)",
                hovermode="x unified",
            )

        elif chart_type == "line":
            fig = go.Figure()
            for comp in components:
                color = CHANNEL_COLORS.get(comp, "#999")
                fig.add_trace(go.Scatter(
                    x=dates, y=dec[comp],
                    name=comp,
                    line=dict(color=color, width=2),
                    mode="lines",
                    hovertemplate=f"{comp}: $%{{y:,.0f}}k<extra></extra>",
                ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Sales Contribution by Component",
                yaxis_title="Sales ($k)",
                hovermode="x unified",
            )

        return fig

    # ── Sales & Spend Trend ──────────────────────────────────────────────────

    def sales_spend_trend(self, chart_type: str = "combo") -> FigOrDf:
        """Time-series trend.  Types: combo | line | bar | table."""
        df    = self.model.df
        dates = df["date"].tolist()

        if chart_type == "table":
            rows = []
            for _, row in df.iterrows():
                rows.append({
                    "Date":          str(row["date"])[:10],
                    "Sales ($k)":    f"${row['sales']:,.0f}k",
                    **{
                        f"{ch} Spend ($k)": f"${row[SPEND_COLS[ch]]:,.0f}k"
                        for ch in MEDIA_CHANNELS
                    },
                })
            return pd.DataFrame(rows)

        if chart_type == "combo":
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for ch in MEDIA_CHANNELS:
                color = CHANNEL_COLORS.get(ch, "#999")
                fig.add_trace(go.Bar(
                    x=dates, y=df[SPEND_COLS[ch]],
                    name=f"{ch} Spend",
                    marker_color=color,
                    opacity=0.8,
                    hovertemplate=f"{ch}: $%{{y:,.0f}}k<extra></extra>",
                ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=dates, y=df["sales"],
                name="Sales",
                line=dict(color="#212121", width=2.5),
                mode="lines",
                hovertemplate="Sales: $%{y:,.0f}k<extra></extra>",
            ), secondary_y=True)
            fig.update_layout(
                **_LAYOUT,
                barmode="stack",
                title=f"{COMPANY_NAME} — Sales & Media Spend Over Time",
                hovermode="x unified",
            )
            fig.update_yaxes(title_text="Media Spend ($k)", secondary_y=False)
            fig.update_yaxes(title_text="Sales ($k)", secondary_y=True)

        elif chart_type == "line":
            fig = go.Figure()
            for ch in MEDIA_CHANNELS:
                color = CHANNEL_COLORS.get(ch, "#999")
                fig.add_trace(go.Scatter(
                    x=dates, y=df[SPEND_COLS[ch]],
                    name=f"{ch} Spend",
                    line=dict(color=color, width=2),
                    mode="lines",
                    hovertemplate=f"{ch}: $%{{y:,.0f}}k<extra></extra>",
                ))
            fig.add_trace(go.Scatter(
                x=dates, y=df["sales"],
                name="Sales",
                line=dict(color="#212121", width=2.5, dash="dot"),
                mode="lines",
                hovertemplate="Sales: $%{y:,.0f}k<extra></extra>",
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Sales & Spend Trends",
                yaxis_title="Value ($k)",
                hovermode="x unified",
            )

        elif chart_type == "bar":
            fig = go.Figure()
            for ch in MEDIA_CHANNELS:
                color = CHANNEL_COLORS.get(ch, "#999")
                fig.add_trace(go.Bar(
                    x=dates, y=df[SPEND_COLS[ch]],
                    name=f"{ch} Spend",
                    marker_color=color,
                    hovertemplate=f"{ch}: $%{{y:,.0f}}k<extra></extra>",
                ))
            fig.update_layout(
                **{**_LAYOUT, "height": 450},
                barmode="stack",
                title=f"{COMPANY_NAME} — Media Spend Over Time",
                yaxis_title="Spend ($k)",
                hovermode="x unified",
            )

        return fig

    # ── Budget Efficiency ────────────────────────────────────────────────────

    def budget_efficiency(self, chart_type: str = "bubble") -> FigOrDf:
        """Spend vs contribution efficiency.  Types: bubble | bar | table."""
        df      = self.model.df
        roi     = self.model.roi
        contribs = self.model.contributions

        spend_totals   = {ch: float(df[SPEND_COLS[ch]].sum()) for ch in MEDIA_CHANNELS}
        contrib_totals = {ch: float(contribs[ch].sum())       for ch in MEDIA_CHANNELS}

        if chart_type == "table":
            return pd.DataFrame([
                {
                    "Channel":                    ch,
                    "Total Spend ($k)":           f"${spend_totals[ch]:,.0f}k",
                    "Total Contribution ($k)":    f"${contrib_totals[ch]:,.0f}k",
                    "ROI":                        f"{roi[ch]:.2f}x",
                    "Efficiency":                 (
                        "High"   if roi[ch] >= 2
                        else ("Medium" if roi[ch] >= 1 else "Low")
                    ),
                }
                for ch in MEDIA_CHANNELS
            ])

        if chart_type == "bubble":
            fig = go.Figure()
            max_x = max(spend_totals.values())
            for ch in MEDIA_CHANNELS:
                x     = spend_totals[ch]
                y     = contrib_totals[ch]
                r     = roi[ch]
                color = CHANNEL_COLORS.get(ch, "#999")
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    name=ch,
                    marker=dict(
                        color=color,
                        size=max(20, r * 30),
                        opacity=0.85,
                        line=dict(color="white", width=2),
                    ),
                    text=[f"  {ch}<br>  ROI: {r:.2f}x"],
                    textposition="top center",
                    hovertemplate=(
                        f"<b>{ch}</b><br>"
                        f"Spend: ${x:,.0f}k<br>"
                        f"Contribution: ${y:,.0f}k<br>"
                        f"ROI: {r:.2f}x<extra></extra>"
                    ),
                ))
            line_x = np.linspace(0, max_x * 1.3, 50).tolist()
            fig.add_trace(go.Scatter(
                x=line_x, y=line_x,
                mode="lines",
                name="Break-even (1:1)",
                line=dict(color="#E53935", dash="dash", width=1.5),
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Budget Efficiency Matrix",
                xaxis_title="Total Media Spend ($k)",
                yaxis_title="Total Sales Contribution ($k)",
            )

        elif chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    name="Total Spend",
                    x=MEDIA_CHANNELS,
                    y=[spend_totals[ch] for ch in MEDIA_CHANNELS],
                    marker_color=[CHANNEL_COLORS.get(ch, "#999") for ch in MEDIA_CHANNELS],
                    hovertemplate="%{x} Spend: $%{y:,.0f}k<extra></extra>",
                ),
                go.Bar(
                    name="Sales Contribution",
                    x=MEDIA_CHANNELS,
                    y=[contrib_totals[ch] for ch in MEDIA_CHANNELS],
                    marker_color=[CHANNEL_COLORS.get(ch, "#999") for ch in MEDIA_CHANNELS],
                    opacity=0.6,
                    hovertemplate="%{x} Contribution: $%{y:,.0f}k<extra></extra>",
                ),
            ])
            fig.update_layout(
                **_LAYOUT,
                barmode="group",
                title=f"{COMPANY_NAME} — Spend vs Contribution by Channel",
                yaxis_title="Value ($k)",
            )

        return fig

    # ── Decomposition Pie / Donut ─────────────────────────────────────────────

    def decomposition_pie(self, chart_type: str = "donut") -> FigOrDf:
        """Sales decomposition.  Types: donut | pie | bar | table."""
        contribs    = self.model.contributions
        total_sales = float(self.model.df["sales"].sum())
        components  = ["Base"] + MEDIA_CHANNELS
        values      = [max(0.0, float(contribs[c].sum())) for c in components]
        colors      = [CHANNEL_COLORS.get(c, "#999") for c in components]
        pcts        = [v / total_sales * 100 for v in values]

        if chart_type == "table":
            return pd.DataFrame([
                {
                    "Component":    comp,
                    "Total ($k)":   f"${v:,.0f}k",
                    "% of Sales":   f"{p:.1f}%",
                }
                for comp, v, p in zip(components, values, pcts)
            ])

        if chart_type in ("donut", "pie"):
            hole = 0.5 if chart_type == "donut" else 0.0
            fig  = go.Figure(go.Pie(
                labels=components,
                values=values,
                hole=hole,
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                texttemplate="%{label}<br>%{percent}",
                hovertemplate="%{label}: $%{value:,.0f}k (%{percent})<extra></extra>",
            ))
            annotations = []
            if chart_type == "donut":
                annotations = [dict(
                    text=f"${total_sales / 1_000:.1f}M",
                    x=0.5, y=0.5,
                    font=dict(size=16, weight="bold"),
                    showarrow=False,
                )]
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Sales Decomposition",
                annotations=annotations,
            )

        elif chart_type == "bar":
            sorted_data = sorted(zip(components, values, pcts), key=lambda x: -x[1])
            chs, vals, ps = zip(*sorted_data)
            fig = go.Figure(go.Bar(
                x=list(chs),
                y=list(vals),
                marker_color=[CHANNEL_COLORS.get(c, "#999") for c in chs],
                text=[f"{p:.1f}%" for p in ps],
                textposition="outside",
                hovertemplate="%{x}: $%{y:,.0f}k (%{text})<extra></extra>",
            ))
            fig.update_layout(
                **_LAYOUT,
                title=f"{COMPANY_NAME} — Sales Decomposition by Component",
                yaxis_title="Sales Contribution ($k)",
            )

        return fig
