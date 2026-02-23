"""
insight_generation_agent.py -- Agent 5: Business Insight Generation

Responsibilities:
  * Translate statistical model outputs into executive-level language
  * Generate ROI analysis, saturation insights, contribution breakdowns
  * Produce budget optimisation recommendations
  * Format Q&A responses for user-posed natural-language questions

Principles:
  * Always lead with the business implication
  * Quantify ROI, contribution %, and dollar impact
  * Flag risks and caveats clearly
  * End with recommended actions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.settings import MEDIA_CHANNELS, SPEND_COLS, CURRENCY_SYMBOL, COMPANY_NAME


class InsightGenerationAgent:
    """
    Converts model outputs into executive-ready business insights.

    Requires a fitted ModelingAgent to be passed at construction.
    """

    # Saturation thresholds (fraction of max response)
    HIGH_SATURATION_THRESHOLD = 0.75    # Channel likely saturated
    LOW_SATURATION_THRESHOLD  = 0.35    # Channel has room to grow

    def __init__(self, modeling_agent) -> None:
        """
        Args:
            modeling_agent: A fitted ModelingAgent instance.
        """
        self.model = modeling_agent

    # == Primary Insight Methods =============================================

    def roi_analysis(self) -> Dict:
        """
        Generate a complete ROI analysis for all channels.

        Returns:
            Dict with narrative, table_data, key_insights, actions.
        """
        roi        = self.model.roi
        df         = self.model.df
        total_spend = sum(df[col].sum() for col in SPEND_COLS.values())

        # Rank channels by ROI
        ranked = sorted(roi.items(), key=lambda x: x[1], reverse=True)
        best_ch, best_roi   = ranked[0]
        worst_ch, worst_roi = ranked[-1]

        table_data = []
        for ch, r in ranked:
            spend_col   = SPEND_COLS[ch]
            total_ch_spend = df[spend_col].sum()
            total_contrib  = self.model.contributions[ch].sum()
            spend_share    = total_ch_spend / total_spend * 100
            table_data.append({
                "Channel":          ch,
                "ROI ($ per $1)":   f"{r:.2f}x",
                "Total Spend ($k)": f"{total_ch_spend:,.0f}",
                "Spend Share":      f"{spend_share:.1f}%",
                "Sales Contribution ($k)": f"{total_contrib:,.0f}",
            })

        narrative = (
            f"**{COMPANY_NAME} Media ROI Summary**\n\n"
            f"Over the analysis period, {best_ch} delivered the highest return on "
            f"investment at **{best_roi:.2f}x** -- meaning every $1 invested in "
            f"{best_ch} generated ${best_roi:.2f} in incremental sales.\n\n"
            f"{worst_ch} was the least efficient channel at **{worst_roi:.2f}x**. "
            f"This gap of {best_roi - worst_roi:.2f}x between best and worst "
            f"performers highlights an opportunity to reallocate budget for "
            f"higher returns."
        )

        key_insights = [
            f"* {best_ch} is your highest-ROI channel ({best_roi:.2f}x) -- protect this budget.",
            f"* {worst_ch} is your least efficient channel ({worst_roi:.2f}x).",
            f"* ROI spread across channels: {worst_roi:.2f}x - {best_roi:.2f}x.",
        ]

        actions = [
            f"1. Protect or increase {best_ch} investment given its superior ROI.",
            f"2. Audit {worst_ch} creative and targeting before next budget cycle.",
            f"3. Conduct incrementality tests on mid-tier channels to validate model ROI.",
        ]

        return {
            "type":         "roi_analysis",
            "narrative":    narrative,
            "table_data":   table_data,
            "key_insights": key_insights,
            "risks":        [
                "ROI estimates assume stable market conditions.",
                "Model uses fixed adstock/saturation parameters -- live calibration would improve precision.",
            ],
            "actions":      actions,
        }

    def saturation_analysis(self) -> Dict:
        """
        Identify which channels are saturated or under-invested.

        Returns:
            Dict with saturation status per channel and narrative.
        """
        curves        = self.model.get_response_curves()
        df            = self.model.df
        channel_status = {}

        for ch in MEDIA_CHANNELS:
            curve         = curves[ch]
            max_lift      = curve["sales_lift"].max()
            current_lift  = curve["current_lift"]
            pct_saturation = current_lift / max_lift if max_lift > 0 else 0.0

            if pct_saturation >= self.HIGH_SATURATION_THRESHOLD:
                status = "SATURATED"
                emoji  = "[HIGH]"
                action = f"Consider reallocating {ch} budget -- marginal returns are very low."
            elif pct_saturation >= 0.50:
                status = "MODERATE"
                emoji  = "[MED]"
                action = f"{ch} is approaching saturation -- incremental investment should be modest."
            else:
                status = "UNDER-INVESTED"
                emoji  = "[LOW]"
                action = f"{ch} has headroom to grow -- increasing spend could yield strong returns."

            channel_status[ch] = {
                "saturation_pct": round(pct_saturation * 100, 1),
                "status":         status,
                "indicator":      emoji,
                "action":         action,
                "current_spend":  round(curve["current_spend"], 1),
            }

        saturated   = [c for c, s in channel_status.items() if s["status"] == "SATURATED"]
        under_inv   = [c for c, s in channel_status.items() if s["status"] == "UNDER-INVESTED"]

        narrative = (
            "**Saturation Analysis -- Diminishing Returns Assessment**\n\n"
            "The saturation percentage shows how close each channel is to its "
            "maximum modelled response. A channel at 80%+ saturation means "
            "additional spend yields very little incremental sales.\n\n"
        )
        if saturated:
            narrative += (
                f"!! **Saturated channels** ({', '.join(saturated)}): "
                f"These are over-invested relative to their response curves. "
                f"Immediate reallocation opportunity.\n\n"
            )
        if under_inv:
            narrative += (
                f"-> **Under-invested channels** ({', '.join(under_inv)}): "
                f"Meaningful upside remains -- consider shifting budget here."
            )

        key_insights = [
            f"* {ch}: {s['saturation_pct']}% saturated -- {s['status']}"
            for ch, s in channel_status.items()
        ]

        return {
            "type":           "saturation_analysis",
            "channel_status": channel_status,
            "narrative":      narrative,
            "key_insights":   key_insights,
            "risks": [
                "Saturation curves are estimated from historical spend ranges -- "
                "extrapolation beyond historical max spend is uncertain.",
            ],
            "actions": [s["action"] for s in channel_status.values()],
        }

    def contribution_analysis(self) -> Dict:
        """
        Break down total sales by channel contribution and organic base.

        Returns:
            Dict with contribution percentages, narrative, and table.
        """
        contribs     = self.model.contributions
        total_sales  = self.model.df["sales"].sum()

        pct = {
            ch: contribs[ch].sum() / total_sales * 100
            for ch in list(MEDIA_CHANNELS) + ["Base"]
        }
        top_media_ch = max(MEDIA_CHANNELS, key=lambda c: pct[c])
        base_pct     = pct["Base"]
        media_pct    = 100 - base_pct

        table_data = sorted(
            [{"Component": k, "Sales Share": f"{v:.1f}%",
              "Sales ($k)": f"{contribs[k].sum():,.0f}"}
             for k, v in pct.items()],
            key=lambda r: -float(r["Sales Share"].strip("%")),
        )

        narrative = (
            f"**Sales Decomposition -- What's Driving Revenue?**\n\n"
            f"Of total sales, **{base_pct:.0f}%** ({CURRENCY_SYMBOL}{contribs['Base'].sum():,.0f}k) "
            f"comes from organic baseline demand -- brand equity, distribution, and "
            f"macro factors independent of paid media.\n\n"
            f"**{media_pct:.0f}%** of sales is attributable to paid media investment. "
            f"{top_media_ch} is the single largest media contributor at "
            f"{pct[top_media_ch]:.1f}% of total sales."
        )

        key_insights = [
            f"* Organic base drives {base_pct:.0f}% of sales -- strong brand foundation.",
            f"* Paid media contributes {media_pct:.0f}% of total revenue.",
            f"* {top_media_ch} is the top media driver ({pct[top_media_ch]:.1f}% of sales).",
        ]

        return {
            "type":         "contribution_analysis",
            "narrative":    narrative,
            "table_data":   table_data,
            "pct":          pct,
            "key_insights": key_insights,
            "risks": [
                "Base sales may include unmeasured factors (PR, distribution changes).",
            ],
            "actions": [
                "1. Protect organic brand-building activities that sustain base sales.",
                f"2. Continue investing in {top_media_ch} -- it is your #1 sales driver.",
                "3. Explore what would happen to base sales if all media were cut.",
            ],
        }

    def budget_scenario(self, channel: str, pct_change: float) -> Dict:
        """
        What-if scenario: change a channel's budget by pct_change.

        Args:
            channel:    Channel name (e.g. 'TV').
            pct_change: e.g. -0.20 for a 20% cut, +0.10 for a 10% increase.

        Returns:
            Scenario analysis dict.
        """
        result   = self.model.simulate_budget_change(channel, pct_change)
        direction = "increase" if pct_change > 0 else "cut"
        sign      = "+" if result["delta_annual_sales"] >= 0 else ""

        narrative = (
            f"**Budget Scenario: {abs(pct_change):.0%} {direction.capitalize()} in {channel}**\n\n"
            f"If {channel} weekly spend moves from "
            f"{CURRENCY_SYMBOL}{result['current_weekly_spend']:,.0f}k -> "
            f"{CURRENCY_SYMBOL}{result['new_weekly_spend']:,.0f}k, "
            f"the model estimates an annual sales impact of "
            f"**{sign}{CURRENCY_SYMBOL}{result['delta_annual_sales']:,.0f}k**.\n\n"
        )

        if abs(pct_change) >= 0.20 and result["delta_annual_sales"] < result["delta_annual_spend"]:
            narrative += (
                f"⚠️  The projected sales gain is *less than* the additional spend, "
                f"suggesting {channel} may be operating in a region of diminishing returns. "
                f"Consider reallocating to a higher-ROI channel instead."
            )

        return {
            "type":      "budget_scenario",
            "scenario":  result,
            "narrative": narrative,
            "key_insights": [
                f"* Annual spend change: {sign}{CURRENCY_SYMBOL}{result['delta_annual_spend']:,.0f}k",
                f"* Estimated annual sales change: {sign}{CURRENCY_SYMBOL}{result['delta_annual_sales']:,.0f}k",
                f"* New {channel} ROI: {result['new_channel_roi']:.2f}x",
            ],
            "risks": [
                "Scenario assumes linear market response -- actual results may vary.",
                "Does not account for competitor responses or media availability.",
            ],
            "actions": [
                f"1. Pilot the {direction} on a regional test market before full rollout.",
                "2. Track actual sales vs model prediction to recalibrate.",
            ],
        }

    def trend_analysis(self, period: Optional[str] = None) -> Dict:
        """
        Analyse sales trends, seasonality, and period-specific drivers.

        Args:
            period: Optional period descriptor (e.g. 'Q4', '2022').

        Returns:
            Trend analysis dict.
        """
        df   = self.model.df
        dec  = self.model.decomposition

        # Quarterly rollup
        quarterly = (
            dec.assign(quarter_label=df["year"].astype(str) + "-Q" + df["quarter"].astype(str))
            .groupby("quarter_label")[["actual", "fitted", "Base"]
            + list(MEDIA_CHANNELS)]
            .sum()
            .round(0)
        )

        best_quarter = quarterly["actual"].idxmax()
        worst_quarter = quarterly["actual"].idxmin()

        seasonal_amplitude = df["seasonality"].max() - df["seasonality"].min()
        trend_gain = (df["trend"].iloc[-1] - df["trend"].iloc[0])  # always 1 here

        narrative = (
            "**Sales Trend & Seasonality Analysis**\n\n"
            f"Sales show a positive long-term trend across the analysis window. "
            f"Seasonality accounts for approximately "
            f"{seasonal_amplitude:.0%} swing in baseline sales between peak "
            f"and trough periods.\n\n"
            f"**Best quarter:** {best_quarter} "
            f"(${quarterly.loc[best_quarter,'actual']:,.0f}k)\n"
            f"**Weakest quarter:** {worst_quarter} "
            f"(${quarterly.loc[worst_quarter,'actual']:,.0f}k)\n\n"
        )

        if period and "Q4" in period.upper():
            q4_rows = quarterly[quarterly.index.str.contains("Q4")]
            if not q4_rows.empty:
                q4_avg = q4_rows["actual"].mean()
                narrative += (
                    f"Q4 performance (avg ${q4_avg:,.0f}k) is typically above-average, "
                    f"driven by seasonal demand uplift and concentrated holiday "
                    f"media campaigns."
                )

        return {
            "type":         "trend_analysis",
            "narrative":    narrative,
            "quarterly":    quarterly.to_dict(),
            "key_insights": [
                f"* Strongest quarter: {best_quarter}",
                f"* Weakest quarter: {worst_quarter}",
                f"* Seasonality causes ~{seasonal_amplitude:.0%} swing in baseline sales.",
            ],
            "risks": [
                "Seasonality estimated from Fourier terms -- external events not captured.",
            ],
            "actions": [
                "1. Front-load media budgets ahead of seasonal peaks.",
                "2. Review media flight plans to align with demand curve.",
            ],
        }

    def executive_summary(self) -> Dict:
        """
        Generate a full executive summary combining all analyses.

        Suitable for a C-suite briefing or investor deck slide deck.
        """
        roi_r  = self.roi_analysis()
        sat_r  = self.saturation_analysis()
        con_r  = self.contribution_analysis()
        fit_q  = self.model.get_fit_quality()

        best_ch  = max(self.model.roi, key=self.model.roi.get)
        best_roi = self.model.roi[best_ch]
        worst_ch = min(self.model.roi, key=self.model.roi.get)

        saturated_chs = [
            ch for ch, s in sat_r["channel_status"].items()
            if s["status"] == "SATURATED"
        ]
        under_chs = [
            ch for ch, s in sat_r["channel_status"].items()
            if s["status"] == "UNDER-INVESTED"
        ]

        headline = (
            f"## {COMPANY_NAME} -- Marketing Mix Model Executive Briefing\n\n"
            f"**Model Accuracy:** R² = {fit_q.get('R²', 'N/A')} | "
            f"MAPE = {fit_q.get('MAPE (%)', 'N/A')}%\n\n"
            "---\n\n"
            "### Key Findings\n\n"
            f"1. **{best_ch} delivers the highest ROI** at {best_roi:.2f}x return per dollar invested.\n"
            f"2. **{worst_ch} is the weakest performer** -- recommend creative or targeting review.\n"
            f"3. **{con_r['pct']['Base']:.0f}%** of sales is organic base demand; "
            f"**{100-con_r['pct']['Base']:.0f}%** is media-driven.\n"
        )
        if saturated_chs:
            headline += (
                f"4. **Saturation alert:** {', '.join(saturated_chs)} "
                f"appear over-invested -- diminishing returns observed.\n"
            )
        if under_chs:
            headline += (
                f"5. **Growth opportunity:** {', '.join(under_chs)} "
                f"have headroom -- consider increasing investment.\n"
            )

        return {
            "type":         "executive_summary",
            "headline":     headline,
            "roi":          roi_r,
            "saturation":   sat_r,
            "contribution": con_r,
            "fit_quality":  fit_q,
            "key_insights": (
                roi_r["key_insights"]
                + sat_r["key_insights"]
                + con_r["key_insights"]
            ),
            "top_actions": [
                f"1. Double down on {best_ch} -- highest proven ROI in the portfolio.",
                f"2. Reallocate budget from {worst_ch} to higher-ROI channels.",
                "3. Align media flights to seasonal demand curve to maximise efficiency.",
                "4. Establish a quarterly MMM refresh to track ROI changes over time.",
            ],
        }
