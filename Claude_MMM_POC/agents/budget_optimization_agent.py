"""
budget_optimization_agent.py — Agent 8: Budget Optimiser

Responsibilities:
  * Find the optimal weekly spend allocation across channels
  * Respect a total budget constraint and per-channel floor / cap
  * Use channel response curves (Hill-function saturation) for accuracy
  * Return recommended reallocation with predicted sales impact

Algorithm:
  1. Pull response curves from ModelingAgent.get_response_curves()
  2. Build scipy.interpolate.interp1d functions (spend -> weekly sales lift)
  3. Run scipy.optimize.minimize (SLSQP, gradient-based nonlinear optimizer)
     Objective  : maximise sum of weekly sales lift across all channels
     Constraints: total weekly spend <= total_budget
     Bounds     : per-channel (min_alloc, max_alloc)
  4. Report optimal allocation vs current allocation

Usage:
    from agents.budget_optimization_agent import BudgetOptimizationAgent

    opt  = BudgetOptimizationAgent(model_agent)
    result = opt.optimize(total_budget=3000.0)   # weekly $000s
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from scipy.interpolate import interp1d
from scipy.optimize import minimize, LinearConstraint, Bounds

from config.settings import MEDIA_CHANNELS, SPEND_COLS


class BudgetOptimizationAgent:
    """
    Optimises weekly media spend allocation to maximise predicted sales.

    Args:
        model: Fitted ModelingAgent instance.

    After calling optimize(), all results are returned as a dict.
    """

    def __init__(self, model) -> None:
        self._model = model

    # == Public Interface ====================================================

    def optimize(
        self,
        total_budget: float,
        min_alloc: Optional[Dict[str, float]] = None,
        max_alloc: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Find the spend allocation that maximises predicted weekly sales.

        Args:
            total_budget: Total weekly budget ($000s) to allocate across all channels.
            min_alloc:    Minimum weekly spend per channel ($000s).
                          Defaults to 0 for each channel.
            max_alloc:    Maximum weekly spend per channel ($000s).
                          Defaults to 3x the channel's current mean spend.

        Returns:
            Dict with keys:
                type               — "budget_optimize"
                optimal_allocation — {channel: weekly_$} at optimum
                current_allocation — {channel: current_mean_weekly_$}
                optimal_sales_lift — total weekly sales lift at optimum ($000s)
                current_sales_lift — total weekly sales lift at current spend ($000s)
                annual_improvement — projected annual gain in sales ($000s)
                improvement_pct    — % improvement over current
                channel_changes    — {channel: "+12%" or "-8%"}
                narrative          — plain-English summary
                table_data         — list of dicts for tabular rendering
                key_insights       — list of bullet strings
                actions            — list of recommended actions
        """
        curves      = self._model.get_response_curves()
        interp_fns  = self._build_interpolators(curves)

        current_alloc = {
            ch: self._model.df[SPEND_COLS[ch]].mean()
            for ch in MEDIA_CHANNELS
        }
        current_lift = sum(
            float(interp_fns[ch](current_alloc[ch])) for ch in MEDIA_CHANNELS
        )

        # Default bounds: 0 to 3x current
        lb = [
            (min_alloc or {}).get(ch, 0.0)
            for ch in MEDIA_CHANNELS
        ]
        ub = [
            (max_alloc or {}).get(ch, max(current_alloc[ch] * 3, 1.0))
            for ch in MEDIA_CHANNELS
        ]
        bounds = Bounds(lb=lb, ub=ub)

        # Constraint: sum of spend <= total_budget
        budget_constraint = LinearConstraint(
            A=np.ones((1, len(MEDIA_CHANNELS))),
            lb=0,
            ub=total_budget,
        )

        # Starting point: scale current allocation to fill budget
        x0 = np.array([current_alloc[ch] for ch in MEDIA_CHANNELS])
        x0 = np.clip(x0, lb, ub)
        x0_sum = x0.sum()
        if x0_sum > 0:
            x0 = x0 * min(total_budget / x0_sum, 1.0)

        result = minimize(
            fun=self._neg_sales(interp_fns),
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=budget_constraint,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        raw_alloc  = {ch: float(result.x[i]) for i, ch in enumerate(MEDIA_CHANNELS)}

        # Clip to guarantee budget constraint is satisfied (handles SLSQP numerical drift)
        raw_total = sum(raw_alloc.values())
        if raw_total > total_budget:
            scale = total_budget / raw_total
            raw_alloc = {ch: v * scale for ch, v in raw_alloc.items()}

        opt_alloc  = raw_alloc
        opt_lift   = sum(float(interp_fns[ch](opt_alloc[ch])) for ch in MEDIA_CHANNELS)

        annual_improvement = (opt_lift - current_lift) * 52
        improvement_pct    = (
            (opt_lift - current_lift) / max(current_lift, 1e-9) * 100
        )

        channel_changes = {}
        for ch in MEDIA_CHANNELS:
            cur = current_alloc[ch]
            opt = opt_alloc[ch]
            delta_pct = (opt - cur) / max(cur, 1e-9) * 100
            channel_changes[ch] = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"

        table_data = self._build_table(
            current_alloc, opt_alloc, channel_changes, interp_fns
        )
        narrative  = self._build_narrative(
            total_budget, opt_alloc, opt_lift, current_lift,
            annual_improvement, improvement_pct
        )
        insights   = self._build_insights(channel_changes, opt_alloc, interp_fns, curves)
        actions    = self._build_actions(channel_changes, opt_alloc, current_alloc)

        return {
            "type":               "budget_optimize",
            "optimal_allocation": opt_alloc,
            "current_allocation": current_alloc,
            "optimal_sales_lift": round(opt_lift, 2),
            "current_sales_lift": round(current_lift, 2),
            "annual_improvement": round(annual_improvement, 1),
            "improvement_pct":    round(improvement_pct, 1),
            "channel_changes":    channel_changes,
            "narrative":          narrative,
            "table_data":         table_data,
            "key_insights":       insights,
            "risks":              [
                "Optimisation assumes historical adstock decay and saturation params hold.",
                "Does not account for competitive dynamics or macro-economic shifts.",
                "Per-channel bounds are required to prevent implausible corner solutions.",
            ],
            "actions":            actions,
            "scenario":           {
                "total_budget_weekly": total_budget,
                "current_total_weekly": sum(current_alloc.values()),
            },
        }

    # == Private Helpers =====================================================

    def _build_interpolators(
        self, curves: Dict[str, Dict]
    ) -> Dict[str, interp1d]:
        """Create interpolation function (spend -> weekly sales lift) per channel."""
        fns = {}
        for ch, c in curves.items():
            spend = np.array(c["spend"])
            lift  = np.array(c["sales_lift"])
            # Linear extrapolation beyond the curve range
            fns[ch] = interp1d(
                spend, lift, kind="linear", fill_value="extrapolate"
            )
        return fns

    def _neg_sales(self, interp_fns: Dict[str, interp1d]):
        """Return the objective function (negate for minimisation)."""
        channels = MEDIA_CHANNELS

        def objective(x: np.ndarray) -> float:
            total = sum(float(interp_fns[ch](x[i])) for i, ch in enumerate(channels))
            return -total

        return objective

    def _build_table(
        self,
        current: Dict[str, float],
        optimal: Dict[str, float],
        changes: Dict[str, str],
        interp_fns: Dict[str, interp1d],
    ) -> List[Dict]:
        rows = []
        for ch in MEDIA_CHANNELS:
            cur_lift = float(interp_fns[ch](current[ch]))
            opt_lift = float(interp_fns[ch](optimal[ch]))
            rows.append({
                "Channel":             ch,
                "Current Spend (wk $k)": f"{current[ch]:.1f}",
                "Optimal Spend (wk $k)": f"{optimal[ch]:.1f}",
                "Change":              changes[ch],
                "Current Sales Lift":  f"{cur_lift:.1f}",
                "Optimal Sales Lift":  f"{opt_lift:.1f}",
            })
        return rows

    def _build_narrative(
        self,
        budget: float,
        opt_alloc: Dict[str, float],
        opt_lift: float,
        cur_lift: float,
        annual_gain: float,
        pct: float,
    ) -> str:
        top_ch = max(opt_alloc, key=opt_alloc.get)
        gain_sign = "+" if annual_gain >= 0 else ""
        return (
            f"With a weekly media budget of ${budget:,.0f}k, the optimizer recommends "
            f"reallocating spend to maximise sales contribution. "
            f"The optimal mix concentrates investment in {top_ch} "
            f"and adjusts other channels based on their marginal return curves. "
            f"Predicted weekly sales lift improves from ${cur_lift:,.1f}k to ${opt_lift:,.1f}k, "
            f"a {gain_sign}{pct:.1f}% uplift translating to {gain_sign}${annual_gain:,.0f}k "
            f"in annual incremental revenue."
        )

    def _build_insights(
        self,
        changes: Dict[str, str],
        opt_alloc: Dict[str, float],
        interp_fns: Dict[str, interp1d],
        curves: Dict[str, Dict],
    ) -> List[str]:
        insights = []
        # Highest-lift channel
        lifts = {ch: float(interp_fns[ch](opt_alloc[ch])) for ch in MEDIA_CHANNELS}
        best  = max(lifts, key=lifts.get)
        insights.append(
            f"* {best} generates the highest incremental sales lift "
            f"(${lifts[best]:,.1f}k/wk) at the recommended spend level."
        )
        # Biggest increases / decreases
        for ch, chg in changes.items():
            if chg.startswith("+") and float(chg[1:-1]) > 10:
                insights.append(
                    f"* Increase {ch} spend ({chg}) — model identifies remaining "
                    f"headroom on its saturation curve."
                )
            elif not chg.startswith("+") and abs(float(chg[:-1])) > 10:
                insights.append(
                    f"* Reduce {ch} spend ({chg}) — channel is near or past its "
                    f"saturation point; marginal return is diminishing."
                )
        return insights

    def _build_actions(
        self,
        changes: Dict[str, str],
        opt: Dict[str, float],
        cur: Dict[str, float],
    ) -> List[str]:
        actions = []
        step = 1
        for ch in MEDIA_CHANNELS:
            delta = opt[ch] - cur[ch]
            if abs(delta) > 0.5:
                verb = "Increase" if delta > 0 else "Reduce"
                actions.append(
                    f"{step}. {verb} weekly {ch} budget from "
                    f"${cur[ch]:,.1f}k to ${opt[ch]:,.1f}k ({changes[ch]})."
                )
                step += 1
        if not actions:
            actions.append("1. Current allocation is close to optimal — no major changes recommended.")
        return actions
