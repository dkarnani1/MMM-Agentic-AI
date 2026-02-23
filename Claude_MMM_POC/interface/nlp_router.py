"""
nlp_router.py — Natural Language Intent Router

Responsibilities:
  * Classify user questions into MMM analysis intents
  * Extract entities (channels, percentages, time periods)
  * Route to the appropriate InsightGenerationAgent method
  * Optionally use the Claude API for more nuanced understanding

Supported Intents:
  roi_analysis       — "What is the ROI of TV?"
  saturation         — "Which channels are saturated?"
  contribution       — "What's driving sales? / decompose sales"
  budget_scenario    — "What if we cut Digital by 20%?"
  trend              — "What happened in Q4? / show me the trend"
  executive_summary  — "Give me an overview / executive summary"
  budget_optimize    — "What is the optimal budget allocation?"
  help               — "help"
  demo               — "demo"
  unknown            — Fallback

Architecture:
  Primary  : Rule-based keyword matching (zero-latency, no API call)
  Optional : Claude API routing (set ANTHROPIC_API_KEY for richer NLU)
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from config.settings import MEDIA_CHANNELS


# =============================================================================
# Intent Definitions
# =============================================================================

INTENTS = {
    "roi_analysis": {
        "keywords": [
            "roi", "return on investment", "return on ad spend", "roas",
            "efficiency", "cost per", "revenue per", "profitable",
            "which channel is best", "best performing", "best channel",
        ],
        "description": "ROI and efficiency analysis by channel",
    },
    "saturation": {
        "keywords": [
            "saturat", "diminishing return", "diminishing", "over-invest",
            "over invest", "marginal", "plateau", "maxed out",
            "headroom", "under-invest", "under invest",
        ],
        "description": "Saturation and diminishing returns analysis",
    },
    "contribution": {
        "keywords": [
            "contribut", "what's driving", "whats driving",
            "decompos", "breakdown", "breakdown of", "break down",
            "attribution", "how much did", "sales driver", "driving sales",
        ],
        "description": "Sales decomposition and channel contribution",
    },
    "budget_scenario": {
        "keywords": [
            "what if", "if we cut", "if we increas", "if we reduce",
            "realloc", "scenario", "what happens if", "budget change",
            "shift budget", "move budget", "increase spend", "decrease spend",
            "cut spend", "boost spend", "increas", "boost",
        ],
        "description": "Budget scenario / what-if simulation",
    },
    "trend": {
        "keywords": [
            "trend", "over time", "q1", "q2", "q3", "q4", "quarter",
            "season", "annual", "monthly", "historical", "trajectory",
            "growth", "decline", "2020", "2021", "2022", "2023",
            "what happened", "why were sales", "why was", "what drove",
            "drove sales", "what happened in",
        ],
        "description": "Trend and seasonality analysis",
    },
    "executive_summary": {
        "keywords": [
            "overview", "summary", "executive", "brief", "full analysis",
            "tell me everything", "all insights", "big picture",
            "highlight", "main finding", "top line",
        ],
        "description": "Full executive summary",
    },
    "help": {
        "keywords": ["help", "what can you", "example", "how do i", "commands"],
        "description": "Usage help",
    },
    "budget_optimize": {
        "keywords": [
            "optimis", "optimiz", "best allocation", "optimal spend",
            "maximis", "maximiz", "best mix", "efficient allocation",
            "optimal budget", "maximize sales", "maximise sales",
            "allocate budget", "best way to spend", "optimal allocation",
        ],
        "description": "Optimal budget allocation using scipy.optimize",
    },
    "demo": {
        "keywords": ["demo", "show demo", "run demo", "full demo"],
        "description": "Run the full demo analysis",
    },
}


# =============================================================================
# Entity Patterns
# =============================================================================

# Percentage: "20%", "20 percent", "twenty percent"
PCT_PATTERN  = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
# Channel names
CH_PATTERN   = re.compile(
    r"\b(" + "|".join(MEDIA_CHANNELS) + r")\b", re.IGNORECASE
)
# Quarter
QTR_PATTERN  = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
# Direction
DIR_PATTERN  = re.compile(
    r"\b(cut|reduce|decrease|lower|drop|increas|boost|grow|raise|double)\b",
    re.IGNORECASE,
)
# Budget dollar amount: "$3000", "3000k", "$3,000"
BUDGET_PATTERN = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*k?\b", re.IGNORECASE
)


# =============================================================================
# Router
# =============================================================================

class NLPRouter:
    """
    Routes natural-language questions to the correct MMM analysis workflow.

    Usage:
        router = NLPRouter()
        intent, entities = router.parse(question)
    """

    def __init__(self, use_claude_api: bool = False) -> None:
        """
        Args:
            use_claude_api: If True AND ANTHROPIC_API_KEY is set,
                            use the Claude API for intent classification.
                            Falls back to rule-based if API unavailable.
        """
        self._use_claude = use_claude_api and bool(os.getenv("ANTHROPIC_API_KEY"))
        self._claude_client = None

        if self._use_claude:
            try:
                import anthropic
                self._claude_client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                self._use_claude = False

    # == Public Interface ====================================================

    def parse(self, question: str) -> Tuple[str, Dict]:
        """
        Classify intent and extract entities from a user question.

        Args:
            question: Raw user input string.

        Returns:
            Tuple of (intent_name, entities_dict).
            entities_dict may contain:
                channel  (str)   — e.g. "TV"
                pct      (float) — e.g. -0.20 for a 20% cut
                quarter  (str)   — e.g. "Q4"
        """
        intent   = self.classify_intent(question)
        entities = self.extract_entities(question)
        return intent, entities

    def classify_intent(self, question: str) -> str:
        """
        Map a question to one of the defined intents.

        Uses Claude API if configured, otherwise rule-based matching.
        """
        if self._use_claude and self._claude_client:
            return self._classify_with_claude(question)
        return self._classify_rule_based(question)

    def extract_entities(self, question: str) -> Dict:
        """Extract structured entities from the user question."""
        entities: Dict = {}

        # Channel
        ch_match = CH_PATTERN.search(question)
        if ch_match:
            matched = ch_match.group(1).capitalize()
            # Normalise e.g. "digital" → "Digital"
            for ch in MEDIA_CHANNELS:
                if ch.lower() == matched.lower():
                    entities["channel"] = ch
                    break

        # Percentage
        pct_match = PCT_PATTERN.search(question)
        if pct_match:
            pct = float(pct_match.group(1)) / 100.0
            # Determine sign from direction keywords
            dir_match = DIR_PATTERN.search(question)
            if dir_match:
                word = dir_match.group(1).lower()
                if any(w in word for w in ("cut", "reduc", "decreas", "lower", "drop")):
                    pct = -pct
            entities["pct"] = pct

        # Quarter
        qtr_match = QTR_PATTERN.search(question)
        if qtr_match:
            entities["quarter"] = f"Q{qtr_match.group(1)}"

        # Budget dollar amount (for budget_optimize intent)
        budget_match = BUDGET_PATTERN.search(question)
        if budget_match:
            raw = budget_match.group(1).replace(",", "")
            entities["total_budget"] = float(raw)

        return entities

    def get_clarifying_question(self, intent: str, entities: Dict) -> Optional[str]:
        """
        Return a clarifying question if entities are ambiguous.

        For example, if intent is budget_scenario but no channel found.
        """
        if intent == "budget_scenario":
            if "channel" not in entities:
                ch_list = ", ".join(MEDIA_CHANNELS)
                return (
                    f"Which channel would you like to simulate? "
                    f"Available: {ch_list}"
                )
            if "pct" not in entities:
                return (
                    f"By what percentage would you like to change {entities['channel']} "
                    f"spend? (e.g. '20% cut' or '10% increase')"
                )
        return None

    # == Private Methods =====================================================

    def _classify_rule_based(self, question: str) -> str:
        """Score each intent by keyword overlap and return the highest."""
        q_lower = question.lower()
        scores: Dict[str, int] = {intent: 0 for intent in INTENTS}

        for intent, config in INTENTS.items():
            for kw in config["keywords"]:
                if kw in q_lower:
                    scores[intent] += 1

        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            return "unknown"
        return best_intent

    def _classify_with_claude(self, question: str) -> str:
        """
        Use the Claude API to classify the intent.

        Falls back to rule-based on any error.
        """
        intent_list = "\n".join(
            f"  - {k}: {v['description']}" for k, v in INTENTS.items()
        )
        prompt = (
            f"You are an intent classifier for a Marketing Mix Model AI system.\n"
            f"Given the following user question, output ONLY the intent name "
            f"(one of the listed intents, nothing else).\n\n"
            f"Available intents:\n{intent_list}\n\n"
            f"User question: {question}\n\n"
            f"Intent:"
        )
        try:
            message = self._claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            intent = message.content[0].text.strip().lower()
            # Validate returned intent
            if intent in INTENTS:
                return intent
        except Exception:
            pass
        # Fallback
        return self._classify_rule_based(question)
