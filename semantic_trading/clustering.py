"""Semantic clustering of prediction markets using OpenAI embeddings + KMeans."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans

from semantic_trading.config import EMBEDDING_MODEL, OPENAI_API_KEY
from semantic_trading.types import ResolvedMarket

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256


def embed_questions(questions: list[str]) -> np.ndarray:
    """Embed market questions using OpenAI text-embedding-3-small."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(questions), EMBED_BATCH_SIZE):
        batch = questions[i : i + EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_embs = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embs)

    return np.array(all_embeddings)


def cluster_markets(
    markets: list[ResolvedMarket],
    *,
    k: int | None = None,
    min_k: int = 2,
) -> dict[int, list[ResolvedMarket]]:
    """
    Cluster markets into topical groups.

    K = floor(N / 10) per the paper, with a minimum of min_k.
    Returns a dict mapping cluster_id -> list of markets.
    """
    n = len(markets)
    if n < min_k:
        return {0: markets}

    if k is None:
        k = max(min_k, math.floor(n / 10))
    k = min(k, n)

    questions = [m.question for m in markets]
    embeddings = embed_questions(questions)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    clusters: dict[int, list[ResolvedMarket]] = defaultdict(list)
    for market, label in zip(markets, labels):
        clusters[int(label)].append(market)

    logger.info("Clustered %d markets into %d clusters (sizes: %s)",
                n, k, [len(v) for v in clusters.values()])
    return dict(clusters)
