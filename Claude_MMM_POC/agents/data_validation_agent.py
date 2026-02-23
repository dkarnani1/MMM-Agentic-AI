"""
data_validation_agent.py — Agent 2: Data Validation

Responsibilities:
  * Check schema completeness
  * Detect missing values, negative numbers, and outliers
  * Verify temporal continuity (no large date gaps)
  * Produce a structured validation report
  * Flag warnings vs hard failures
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Any

from config.settings import MEDIA_CHANNELS, SPEND_COLS


class DataValidationAgent:
    """
    Validates the raw MMM dataset before feature engineering.

    Call validate(df) → returns a validation report dict.
    The report contains:
        is_valid (bool)   — False means the pipeline should halt
        errors   (list)   — Hard failures
        warnings (list)   — Issues to flag but not stop on
        stats    (dict)   — Summary statistics
    """

    # Thresholds
    MAX_MISSING_PCT   = 0.05   # >5% missing = error
    OUTLIER_STD       = 4.0    # Values beyond ±4σ flagged
    MAX_GAP_WEEKS     = 4      # Gaps larger than this flagged as warnings

    def __init__(self) -> None:
        self.report: Dict[str, Any] = {}

    # == Public Interface ====================================================

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on the dataset.

        Args:
            df: Raw DataFrame from DataIngestionAgent.

        Returns:
            Validation report dictionary.
        """
        errors:   List[str] = []
        warnings: List[str] = []

        errors   += self._check_required_columns(df)
        errors   += self._check_missing_values(df)
        errors   += self._check_negative_values(df)
        warnings += self._check_outliers(df)
        warnings += self._check_date_gaps(df)
        warnings += self._check_zero_spend_weeks(df)

        stats = self._compute_stats(df)

        self.report = {
            "is_valid": len(errors) == 0,
            "errors":   errors,
            "warnings": warnings,
            "stats":    stats,
        }
        return self.report

    def print_report(self) -> None:
        """Print a human-readable validation report to stdout."""
        if not self.report:
            print("No validation has been run yet.")
            return

        status = "[PASS]" if self.report["is_valid"] else "[FAIL]"
        print(f"\n{'='*60}")
        print(f"  DATA VALIDATION REPORT  {status}")
        print(f"{'='*60}")

        if self.report["errors"]:
            print("\n[ERRORS]")
            for e in self.report["errors"]:
                print(f"  x {e}")

        if self.report["warnings"]:
            print("\n[WARNINGS]")
            for w in self.report["warnings"]:
                print(f"  ! {w}")

        if not self.report["errors"] and not self.report["warnings"]:
            print("  All checks passed — no issues found.")

        stats = self.report.get("stats", {})
        if stats:
            print("\n[SUMMARY STATISTICS]")
            for k, v in stats.items():
                print(f"  {k:<25}: {v}")

        print(f"{'='*60}\n")

    # == Private Checks ======================================================

    def _check_required_columns(self, df: pd.DataFrame) -> List[str]:
        required = ["date", "sales"] + list(SPEND_COLS.values())
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return [f"Missing required columns: {missing}"]
        return []

    def _check_missing_values(self, df: pd.DataFrame) -> List[str]:
        errors = []
        numeric_cols = ["sales"] + list(SPEND_COLS.values())
        for col in numeric_cols:
            if col not in df.columns:
                continue
            pct = df[col].isna().mean()
            if pct > self.MAX_MISSING_PCT:
                errors.append(
                    f"Column '{col}' has {pct:.1%} missing values "
                    f"(threshold: {self.MAX_MISSING_PCT:.0%})"
                )
        return errors

    def _check_negative_values(self, df: pd.DataFrame) -> List[str]:
        errors = []
        cols_to_check = ["sales"] + list(SPEND_COLS.values())
        for col in cols_to_check:
            if col not in df.columns:
                continue
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                errors.append(
                    f"Column '{col}' contains {n_neg} negative values."
                )
        return errors

    def _check_outliers(self, df: pd.DataFrame) -> List[str]:
        warnings = []
        cols = ["sales"] + list(SPEND_COLS.values())
        for col in cols:
            if col not in df.columns:
                continue
            z = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
            n_out = (z > self.OUTLIER_STD).sum()
            if n_out > 0:
                warnings.append(
                    f"Column '{col}' has {n_out} outlier(s) "
                    f"beyond {self.OUTLIER_STD}σ — review before modeling."
                )
        return warnings

    def _check_date_gaps(self, df: pd.DataFrame) -> List[str]:
        if "date" not in df.columns:
            return []
        dates  = pd.to_datetime(df["date"]).sort_values()
        deltas = dates.diff().dropna()
        max_gap_days = deltas.max().days
        if max_gap_days > self.MAX_GAP_WEEKS * 7:
            return [
                f"Largest date gap is {max_gap_days} days "
                f"({max_gap_days // 7} weeks). "
                "Check for missing weeks in the data."
            ]
        return []

    def _check_zero_spend_weeks(self, df: pd.DataFrame) -> List[str]:
        warnings = []
        for ch, col in SPEND_COLS.items():
            if col not in df.columns:
                continue
            n_zero = (df[col] == 0).sum()
            pct = n_zero / len(df)
            if pct > 0.20:
                warnings.append(
                    f"{ch}: {n_zero} weeks ({pct:.0%}) with zero spend — "
                    "may affect adstock estimation."
                )
        return warnings

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, str]:
        stats = {
            "Weeks":         str(len(df)),
            "Date range":    f"{df['date'].min().date()} to {df['date'].max().date()}"
                             if "date" in df.columns else "N/A",
            "Avg weekly sales": f"${df['sales'].mean():,.0f}k"
                                if "sales" in df.columns else "N/A",
        }
        for ch, col in SPEND_COLS.items():
            if col in df.columns:
                stats[f"Total {ch} spend"] = f"${df[col].sum():,.0f}k"
        return stats
