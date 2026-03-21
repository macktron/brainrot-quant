#!/usr/bin/env python3
"""
Semantic Trading Live Runner (MVP)

Continuously monitors Polymarket for trading opportunities:
1. Fetch active (unresolved) binary markets
2. Cluster and discover relationships
3. Monitor for leader market resolution
4. Log signals when leaders resolve (optionally execute trades)

Usage:
    python run_live.py                    # monitor mode (signals only)
    python run_live.py --poll-interval 60 # check every 60 seconds
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from semantic_trading.clustering import cluster_markets
from semantic_trading.config import DATA_DIR, ENTRY_PRICE_CUTOFF, GAMMA_API_BASE
from semantic_trading.data import fetch_active_markets, fetch_price_history, get_terminal_price
from semantic_trading.discovery import discover_all_relations
from semantic_trading.labeling import label_all_clusters
from semantic_trading.types import MarketRelation, ResolvedMarket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


def _check_market_resolved(condition_id: str) -> dict | None:
    """Check if a market has resolved via Gamma API."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{GAMMA_API_BASE}/markets/{condition_id}")
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("closed") and data.get("outcome"):
                return data
    except Exception:
        pass
    return None


def _determine_signal(
    relation: MarketRelation,
    leader: ResolvedMarket,
    leader_outcome: str,
) -> dict:
    """Determine trade signal when leader resolves."""
    leader_yes = leader_outcome.lower() == "yes"
    buy_yes = (leader_yes and relation.is_same_outcome) or (
        not leader_yes and not relation.is_same_outcome
    )
    return {
        "side": "YES" if buy_yes else "NO",
        "leader_question": leader.question,
        "leader_outcome": leader_outcome,
        "follower_question": (
            relation.question_j
            if relation.question_i == leader.question
            else relation.question_i
        ),
        "is_same_outcome": relation.is_same_outcome,
        "confidence": relation.confidence_score,
        "rationale": relation.rationale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_discovery_pass(markets: list[ResolvedMarket]) -> list[MarketRelation]:
    """Run the full agentic pipeline on active markets."""
    logger.info("Running discovery on %d active markets...", len(markets))
    clusters = cluster_markets(markets)
    labels = label_all_clusters(clusters)
    relations = discover_all_relations(clusters, labels)
    logger.info("Discovered %d relations across %d clusters", len(relations), len(clusters))
    return relations


def monitor_loop(
    relations: list[MarketRelation],
    markets: list[ResolvedMarket],
    poll_interval: int = 300,
):
    """Poll for leader resolution and emit signals."""
    market_by_question: dict[str, ResolvedMarket] = {m.question: m for m in markets}
    resolved_leaders: set[str] = set()

    pairs: list[tuple[MarketRelation, ResolvedMarket, ResolvedMarket]] = []
    for rel in relations:
        mi = market_by_question.get(rel.question_i)
        mj = market_by_question.get(rel.question_j)
        if mi and mj:
            pairs.append((rel, mi, mj))

    logger.info("Monitoring %d pairs for leader resolution...", len(pairs))

    while pairs:
        for rel, mi, mj in pairs:
            for leader, follower in [(mi, mj), (mj, mi)]:
                if leader.condition_id in resolved_leaders:
                    continue
                result = _check_market_resolved(leader.condition_id)
                if result and result.get("outcome"):
                    resolved_leaders.add(leader.condition_id)
                    signal = _determine_signal(rel, leader, result["outcome"])
                    logger.info(
                        "SIGNAL: %s %s (confidence=%.2f)",
                        signal["side"],
                        signal["follower_question"][:60],
                        signal["confidence"],
                    )
                    sig_path = SIGNALS_DIR / f"signal_{int(time.time())}.json"
                    with open(sig_path, "w") as f:
                        json.dump(signal, f, indent=2)
                    print(f"\n{'='*60}")
                    print(f"  TRADE SIGNAL")
                    print(f"  Side:       {signal['side']}")
                    print(f"  Follower:   {signal['follower_question']}")
                    print(f"  Confidence: {signal['confidence']:.2f}")
                    print(f"  Rationale:  {signal['rationale']}")
                    print(f"  Saved to:   {sig_path}")
                    print(f"{'='*60}\n")

        # Remove fully resolved pairs
        pairs = [
            (rel, mi, mj)
            for rel, mi, mj in pairs
            if mi.condition_id not in resolved_leaders
            or mj.condition_id not in resolved_leaders
        ]

        if pairs:
            logger.info("Still monitoring %d pairs, sleeping %ds...", len(pairs), poll_interval)
            time.sleep(poll_interval)

    logger.info("All monitored pairs resolved or exhausted.")


def main():
    parser = argparse.ArgumentParser(description="Semantic Trading Live Runner")
    parser.add_argument("--max-markets", type=int, default=200,
                        help="Max active markets to fetch (default: 200)")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between resolution checks (default: 300)")
    args = parser.parse_args()

    logger.info("Fetching active markets...")
    markets = fetch_active_markets(limit=args.max_markets)
    logger.info("Found %d active binary markets", len(markets))

    if len(markets) < 5:
        logger.error("Too few active markets (%d). Try increasing --max-markets.", len(markets))
        return

    relations = run_discovery_pass(markets)
    if not relations:
        logger.warning("No relations discovered. Exiting.")
        return

    print(f"\nDiscovered {len(relations)} relations. Entering monitoring loop.\n")
    for i, rel in enumerate(relations, 1):
        direction = "SAME" if rel.is_same_outcome else "DIFF"
        print(f"  {i}. [{direction} {rel.confidence_score:.2f}] "
              f"{rel.question_i[:40]}... <-> {rel.question_j[:40]}...")

    print()
    monitor_loop(relations, markets, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
