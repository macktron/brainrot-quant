"""Cluster labeling via GPT-4o structured output — exact prompt from paper Appendix A."""

from __future__ import annotations

import logging

from openai import OpenAI

from semantic_trading.config import CLUSTER_CATEGORIES, LLM_MODEL, OPENAI_API_KEY
from semantic_trading.types import ClusterLabel, ResolvedMarket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert analyst of prediction markets. "
    "Assign the cluster of markets to one of the following categories: "
    f"{', '.join(repr(c) for c in CLUSTER_CATEGORIES)}. "
    "You must assign exactly one category."
)


def label_cluster(markets: list[ResolvedMarket]) -> str:
    """Assign a category label to a cluster of markets."""
    questions_text = "\n".join(f"- {m.question}" for m in markets)
    user_msg = f"Here are the market questions in this cluster:\n{questions_text}"

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=ClusterLabel,
        temperature=0.0,
    )
    result = resp.choices[0].message.parsed
    category = result.category if result else "other"
    logger.info("Cluster labeled as '%s' (%d markets)", category, len(markets))
    return category


def label_all_clusters(
    clusters: dict[int, list[ResolvedMarket]],
) -> dict[int, str]:
    """Label every cluster, returns cluster_id -> category."""
    labels: dict[int, str] = {}
    for cid, markets in clusters.items():
        labels[cid] = label_cluster(markets)
    return labels
