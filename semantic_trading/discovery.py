"""Relationship discovery via GPT-4o structured output — exact prompt from paper Appendix A."""

from __future__ import annotations

import logging

from openai import OpenAI

from semantic_trading.config import CONFIDENCE_THRESHOLD, LLM_MODEL, ONLY_SAME_OUTCOME, OPENAI_API_KEY
from semantic_trading.types import MarketRelation, MarketRelationList, ResolvedMarket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert prediction-market analyst. Given a list of bets expressed as questions, \
find pairs whose outcomes are LOGICALLY DETERMINED by the same underlying real-world event.

GOOD relationships (propose these):
- Near-duplicate questions: "Will X happen by Nov 30?" and "Will X happen in November?" → same outcome
- Logical implications: "Will tariffs increase?" and "Will tariffs be above 25%?" → if tariffs above 25%, they increased
- Causal chains with high confidence: "Will Trump impose tariffs on EU alcohol?" and "Will the EU impose retaliatory tariffs?" → same outcome (retaliation follows action)
- Logically contradictory: "Will Trump increase tariffs on Canada?" and "Will Trump remove tariffs on Canada?" → different outcome

BAD relationships (do NOT propose these):
- Markets that are merely about the same TOPIC but have independent outcomes
- Two different companies/people/teams doing the same thing independently
- Markets with different time horizons unless one logically implies the other
- Speculative causal links without strong logical basis
- SAME PERSON/ENTITY doing DIFFERENT INDEPENDENT ACTIONS (e.g., "Will Trump cut tariffs?" and "Will Trump cut taxes?" are INDEPENDENT policy decisions — do NOT pair them)
- Markets about different asset price targets at very different times
- "Between X and Y" (narrow range) markets paired with "above/below Z" (threshold) markets — these have entirely different resolution mechanics
- Action and its CONSEQUENCE as "same outcome" (e.g., "Biden endorse Kamala" and "Kamala drop out" are OPPOSITES, not same)
- Markets about what someone will SAY/MENTION at an event: different speech topics are INDEPENDENT (e.g., "Will X say Polymarket?" and "Will X say Youtube?" are independent word choices — do NOT pair them)
- Cumulative count thresholds at different dates (e.g., "1300+ cases by July" and "1500+ cases by September"): counts can accelerate, so NOT hitting a threshold early does NOT guarantee missing a later threshold
- POINT-IN-TIME vs ANY-TIME-IN-PERIOD price markets: "above $X on [specific date]" and "reach $Y in [month]" have different mechanics — failing a specific-date check does NOT mean failing an any-time-in-month check, as prices can spike after that date

CRITICAL: Most prediction markets resolve to "No". For topically related markets, \
"same outcome" (both No) is the most likely relationship. Only predict "different outcome" \
when there is a STRONG logical contradiction.

For each pair you propose:
- question_i and question_j: Output the questions EXACTLY as given.
- is_same_outcome: true if outcomes should match (both yes or both no), false ONLY for logical contradictions.
- confidence_score: 0 to 1. Use 0.9+ only for near-duplicates. Use 0.7-0.9 for strong causal links.
- rationale: Explain the specific logical/causal mechanism linking the outcomes.

Be HIGHLY selective. Only propose pairs where you would bet real money on the relationship."""


def discover_relations(
    markets: list[ResolvedMarket],
    *,
    category: str = "",
) -> list[MarketRelation]:
    """
    Discover same/different-outcome relationships within a cluster of markets.
    Returns relations filtered to confidence >= CONFIDENCE_THRESHOLD.
    """
    if len(markets) < 2:
        return []

    questions_text = "\n".join(f"- {m.question}" for m in markets)

    context = f"Cluster category: {category}\n\n" if category else ""
    user_msg = f"{context}Market questions:\n{questions_text}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=MarketRelationList,
        temperature=0.0,
    )

    result = resp.choices[0].message.parsed
    if result is None:
        return []

    filtered = [r for r in result.relations if r.confidence_score >= CONFIDENCE_THRESHOLD]
    if ONLY_SAME_OUTCOME:
        filtered = [r for r in filtered if r.is_same_outcome]
    # Remove self-matches
    filtered = [
        r for r in filtered
        if r.question_i.strip().lower() != r.question_j.strip().lower()
    ]
    logger.info(
        "Discovered %d relations (%d after confidence filter) in cluster of %d markets",
        len(result.relations), len(filtered), len(markets),
    )
    return filtered


SKIP_CATEGORIES = {"sports"}


def discover_all_relations(
    clusters: dict[int, list[ResolvedMarket]],
    labels: dict[int, str],
) -> list[MarketRelation]:
    """Run relationship discovery across all clusters, skipping sports."""
    all_relations: list[MarketRelation] = []
    for cid, markets in clusters.items():
        category = labels.get(cid, "")
        if category in SKIP_CATEGORIES:
            logger.info("Skipping %s cluster %d (%d markets)", category, cid, len(markets))
            continue
        relations = discover_relations(markets, category=category)
        all_relations.extend(relations)
    logger.info("Total discovered relations: %d", len(all_relations))
    return all_relations
