#!/usr/bin/env python3
"""
Semantic Trading — Daily Pipeline

Runs as a single-shot job (designed for GitHub Actions cron):
1. Fetch active markets + recently resolved markets
2. Cluster and discover relationships
3. For any pair where the leader has already resolved: generate trade signal
4. Execute trade on follower market (if not dry run)
5. Notify via Discord

Usage:
    python run_live.py                      # dry run (signals only, notifications sent)
    python run_live.py --live               # live trading (executes real trades)
    python run_live.py --trade-size 10      # override USDC amount per trade
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from semantic_trading.backtest import _is_date_resolution_mismatch
from semantic_trading.clustering import cluster_markets
from semantic_trading.config import (
    CONFIDENCE_THRESHOLD,
    DATA_DIR,
    DRY_RUN,
    ENTRY_PRICE_CUTOFF,
    GAMMA_API_BASE,
    TRADE_SIZE_USDC,
)
from semantic_trading.data import (
    fetch_active_markets,
    fetch_recently_resolved_markets,
)
from semantic_trading.discovery import discover_all_relations
from semantic_trading.execute import TradeExecution, execute_trade
from semantic_trading.labeling import label_all_clusters
from semantic_trading.notify import send_summary_notification, send_trade_notification
from semantic_trading.types import MarketRelation, ResolvedMarket

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SIGNALS_DIR = DATA_DIR / "signals"
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


def _check_market_resolved(condition_id: str, slug: str = "") -> dict | None:
    """Check if a specific market has resolved via Gamma API."""
    try:
        with httpx.Client(timeout=15.0) as client:
            # Try slug-based lookup first, then conditionId
            for params in [
                {"slug": slug, "limit": "1"} if slug else None,
                {"conditionId": condition_id, "limit": "1"},
            ]:
                if params is None:
                    continue
                resp = client.get(f"{GAMMA_API_BASE}/markets", params=params)
                if resp.status_code != 200:
                    continue
                raw = resp.json()
                if not raw:
                    continue
                data = raw[0] if isinstance(raw, list) else raw
                if not data or not isinstance(data, dict):
                    continue

                if not data.get("closed"):
                    return None  # Market exists but not closed

                outcome = data.get("outcome", "")
                if outcome:
                    return data

                prices_str = data.get("outcomePrices", "")
                try:
                    prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                except (json.JSONDecodeError, TypeError):
                    continue
                if prices and len(prices) == 2:
                    p0, p1 = float(prices[0]), float(prices[1])
                    outcomes_str = data.get("outcomes", "")
                    try:
                        outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if p0 > 0.9:
                        data["outcome"] = outcomes[0]
                        return data
                    if p1 > 0.9:
                        data["outcome"] = outcomes[1]
                        return data
                return None  # Market exists, closed, but no clear outcome
    except Exception:
        pass
    return None


def _determine_signal(
    relation: MarketRelation,
    leader: ResolvedMarket,
    leader_outcome: str,
    follower: ResolvedMarket,
) -> dict:
    """Determine trade signal when leader resolves."""
    leader_yes = leader_outcome.lower() == "yes"
    buy_yes = (leader_yes and relation.is_same_outcome) or (
        not leader_yes and not relation.is_same_outcome
    )
    side = "YES" if buy_yes else "NO"
    token_id = follower.yes_token_id if buy_yes else follower.no_token_id

    return {
        "side": side,
        "token_id": token_id or "",
        "leader_question": leader.question,
        "leader_outcome": leader_outcome,
        "follower_question": follower.question,
        "follower_condition_id": follower.condition_id,
        "is_same_outcome": relation.is_same_outcome,
        "confidence": relation.confidence_score,
        "rationale": relation.rationale,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(
    *,
    max_markets: int = 300,
    dry_run: bool = True,
    trade_size: float = 5.0,
) -> None:
    """Run the full single-shot pipeline."""

    # --- Step 1: Fetch markets ---
    logger.info("Step 1: Fetching markets...")
    active = fetch_active_markets(limit=max_markets)
    recently_resolved = fetch_recently_resolved_markets(days_back=3, limit=200)
    all_markets = active + recently_resolved

    seen_ids = set()
    deduped: list[ResolvedMarket] = []
    for m in all_markets:
        if m.condition_id not in seen_ids:
            seen_ids.add(m.condition_id)
            deduped.append(m)

    logger.info("  Active: %d, Recently resolved: %d, Combined (deduped): %d",
                len(active), len(recently_resolved), len(deduped))

    if len(deduped) < 5:
        logger.error("Too few markets (%d). Exiting.", len(deduped))
        send_summary_notification(
            markets_fetched=len(deduped),
            relations_discovered=0,
            trades_executed=0,
            trades_failed=0,
            dry_run=dry_run,
        )
        return

    # --- Step 2: Cluster + Label + Discover ---
    logger.info("Step 2: Running discovery pipeline...")
    clusters = cluster_markets(deduped)
    labels = label_all_clusters(clusters)
    relations = discover_all_relations(clusters, labels)
    logger.info("  Clusters: %d, Relations: %d", len(clusters), len(relations))

    if not relations:
        logger.info("No relations discovered. Exiting.")
        send_summary_notification(
            markets_fetched=len(deduped),
            relations_discovered=0,
            trades_executed=0,
            trades_failed=0,
            dry_run=dry_run,
        )
        return

    # --- Step 3: Check for tradeable signals ---
    logger.info("Step 3: Checking for tradeable signals...")
    active_ids = {m.condition_id for m in active}
    market_by_question: dict[str, ResolvedMarket] = {m.question: m for m in deduped}

    trades_executed = 0
    trades_failed = 0

    for rel in relations:
        mi = market_by_question.get(rel.question_i)
        mj = market_by_question.get(rel.question_j)
        if not mi or not mj:
            continue

        if _is_date_resolution_mismatch(rel.question_i, rel.question_j):
            continue

        # Check each market as potential leader
        for leader_candidate, follower_candidate in [(mi, mj), (mj, mi)]:
            if follower_candidate.condition_id not in active_ids:
                continue

            resolved_data = _check_market_resolved(
                leader_candidate.condition_id, slug=leader_candidate.market_slug
            )
            if not resolved_data or not resolved_data.get("outcome"):
                continue

            leader_outcome = resolved_data["outcome"]
            signal = _determine_signal(rel, leader_candidate, leader_outcome, follower_candidate)

            if not signal["token_id"]:
                logger.warning("No token_id for follower, skipping")
                continue

            logger.info("SIGNAL: BUY %s on '%s' (confidence=%.2f)",
                        signal["side"], signal["follower_question"][:60], signal["confidence"])

            # Save signal to disk
            sig_path = SIGNALS_DIR / f"signal_{int(time.time())}.json"
            with open(sig_path, "w") as f:
                json.dump(signal, f, indent=2)

            # Execute trade
            execution = TradeExecution(success=False, error="Dry run")
            if not dry_run:
                execution = execute_trade(
                    token_id=signal["token_id"],
                    amount_usdc=trade_size,
                )
                if execution.success:
                    trades_executed += 1
                    logger.info("Trade executed: order=%s", execution.order_id)
                else:
                    trades_failed += 1
                    logger.error("Trade failed: %s", execution.error)

            # Notify
            send_trade_notification(
                side=signal["side"],
                follower_question=signal["follower_question"],
                leader_question=signal["leader_question"],
                leader_outcome=leader_outcome,
                confidence=signal["confidence"],
                rationale=signal["rationale"],
                amount_usdc=trade_size if not dry_run else None,
                order_id=execution.order_id,
                error=execution.error if not execution.success and not dry_run else None,
            )

            time.sleep(1)

    # --- Step 4: Summary ---
    logger.info("Pipeline complete. Executed: %d, Failed: %d", trades_executed, trades_failed)
    send_summary_notification(
        markets_fetched=len(deduped),
        relations_discovered=len(relations),
        trades_executed=trades_executed,
        trades_failed=trades_failed,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="Semantic Trading — Daily Pipeline")
    parser.add_argument("--max-markets", type=int, default=300,
                        help="Max active markets to fetch (default: 300)")
    parser.add_argument("--live", action="store_true",
                        help="Enable live trading (default: dry run)")
    parser.add_argument("--trade-size", type=float, default=None,
                        help=f"USDC per trade (default: ${TRADE_SIZE_USDC})")
    args = parser.parse_args()

    is_dry_run = DRY_RUN and not args.live
    size = args.trade_size or TRADE_SIZE_USDC

    mode = "LIVE" if not is_dry_run else "DRY RUN"
    logger.info("Starting Semantic Trading pipeline (%s, $%.2f/trade)", mode, size)

    run_pipeline(
        max_markets=args.max_markets,
        dry_run=is_dry_run,
        trade_size=size,
    )


if __name__ == "__main__":
    main()
