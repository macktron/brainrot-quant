"""Relationship discovery via GPT-4o structured output — exact prompt from paper Appendix A."""

from __future__ import annotations

import logging

from openai import OpenAI

from semantic_trading.config import CONFIDENCE_THRESHOLD, LLM_MODEL, OPENAI_API_KEY
from semantic_trading.types import MarketRelation, MarketRelationList, ResolvedMarket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert prediction-market analyst. Given a list of bets expressed as questions, \
find pairs of bets whose outcomes are very likely to be related to each other.

For each pair you propose, create a new MarketRelation and fill out the following:
- In the question_i and question_j fields, output the questions in the relationship. \
Output the questions exactly as they are given.
- In the is_same_outcome field, output true if outcomes are likely to be the same \
(both yes or both no), false if outcomes are likely to be different (one yes and one no). \
You must provide a boolean.
- In the confidence_score field, provide a score between 0 and 1 indicating how confident \
you are about the relationship. You must provide a confidence score.
- In the rationale field, justify why you chose these bets and the relationship you \
assigned to them. You must provide a rationale.

Only propose pairs whose outcomes are very likely related. Be selective and precise."""


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
        temperature=0.7,
    )

    result = resp.choices[0].message.parsed
    if result is None:
        return []

    filtered = [r for r in result.relations if r.confidence_score >= CONFIDENCE_THRESHOLD]
    logger.info(
        "Discovered %d relations (%d after confidence filter) in cluster of %d markets",
        len(result.relations), len(filtered), len(markets),
    )
    return filtered


def discover_all_relations(
    clusters: dict[int, list[ResolvedMarket]],
    labels: dict[int, str],
) -> list[MarketRelation]:
    """Run relationship discovery across all clusters."""
    all_relations: list[MarketRelation] = []
    for cid, markets in clusters.items():
        category = labels.get(cid, "")
        relations = discover_relations(markets, category=category)
        all_relations.extend(relations)
    logger.info("Total discovered relations: %d", len(all_relations))
    return all_relations
