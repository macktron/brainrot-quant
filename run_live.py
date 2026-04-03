#!/usr/bin/env python3
"""
Semantic Trading — Daily Pipeline

Scheduled runs (GitHub Actions cron): LIVE trading with real money.
Manual triggers (workflow_dispatch):  Paper trading only.

Position sizing is automatic — fetches current USDC balance from Polymarket
and sizes each bet as ~20% of available capital (confidence-scaled).

Usage:
    python run_live.py                      # uses DRY_RUN env var (default: true)
    python run_live.py --live               # force live trading
    python run_live.py --paper              # force paper trading
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone

import httpx

from semantic_trading.backtest import _is_date_resolution_mismatch
from semantic_trading.clustering import cluster_markets
from semantic_trading.config import (
    DATA_DIR,
    DRY_RUN,
    GAMMA_API_BASE,
    MAX_TRADES_PER_RUN,
    SKIP_EXISTING_POSITIONS,
)
from semantic_trading.data import (
    fetch_active_markets,
    fetch_recently_resolved_markets,
)
from semantic_trading.discovery import discover_all_relations
from semantic_trading.execute import (
    BalanceInfo,
    TradeExecution,
    compute_trade_size,
    execute_trade,
    fetch_balance,
)
from semantic_trading.exposure import (
    ExposureInfo,
    filter_markets_by_exposure,
    load_full_exposure,
)
from semantic_trading.history import record_trade, save_run
from semantic_trading.labeling import label_all_clusters
from semantic_trading.notify import (
    send_balance_alert,
    send_error_alert,
    send_summary_notification,
    send_trade_notification,
)
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
                    return None

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
                return None
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


def _normalize_question(q: str) -> str:
    """Normalize question text for robust matching with LLM output."""
    q = q.strip()
    # GPT sometimes echoes questions as bullet items.
    if q.startswith("- "):
        q = q[2:].strip()
    # Mirror backtester logic for removing date suffixes.
    q = re.sub(r"\s*\(start:.*$", "", q)
    return q.strip().lower()


def _resolve_market_by_question(
    question: str,
    markets: list[ResolvedMarket],
    *,
    market_by_question_exact: dict[str, ResolvedMarket],
    market_by_question_norm: dict[str, ResolvedMarket],
    normalized_market_keys: list[str],
) -> ResolvedMarket | None:
    """Resolve a market from an LLM-provided question string."""
    if question in market_by_question_exact:
        return market_by_question_exact[question]

    nq = _normalize_question(question)
    if nq in market_by_question_norm:
        return market_by_question_norm[nq]

    # Last resort: substring match on normalized strings.
    for mk in normalized_market_keys:
        if nq in mk or mk in nq:
            return market_by_question_norm.get(mk)

    return None


def run_pipeline(*, max_markets: int = 300, dry_run: bool = True) -> None:
    """Run the full single-shot pipeline with dynamic position sizing."""

    # --- Step 0: Check balance (live mode only) ---
    balance = BalanceInfo(balance_usdc=0.0, is_bankrupt=False, is_low=False)
    if not dry_run:
        logger.info("Step 0: Checking balance...")
        balance = fetch_balance()
        logger.info("  Balance: $%.2f USDC", balance.balance_usdc)

        if balance.is_bankrupt:
            logger.error("BANKRUPT: $%.2f — halting all trading", balance.balance_usdc)
            send_balance_alert(balance_usdc=balance.balance_usdc, is_bankrupt=True)
            return

        if balance.is_low:
            logger.warning("LOW BALANCE: $%.2f — trades will be small", balance.balance_usdc)
            send_balance_alert(balance_usdc=balance.balance_usdc, is_bankrupt=False)

    # --- Step 0b: Load existing exposure ---
    exposure = ExposureInfo()
    if not dry_run:
        logger.info("Step 0b: Loading existing exposure...")
        exposure = load_full_exposure(include_api_positions=True)
        if exposure.positions:
            logger.info(
                "  Current exposure: %d positions, $%.2f total",
                len(exposure.positions),
                exposure.total_exposure_usdc,
            )
        else:
            logger.info("  No existing positions found")

    # --- Step 1: Fetch markets ---
    logger.info("Step 1: Fetching markets...")
    active = fetch_active_markets(limit=max_markets)
    recently_resolved = fetch_recently_resolved_markets(days_back=3, limit=200)
    all_markets = active + recently_resolved

    seen_ids: set[str] = set()
    deduped: list[ResolvedMarket] = []
    for m in all_markets:
        if m.condition_id not in seen_ids:
            seen_ids.add(m.condition_id)
            deduped.append(m)

    logger.info("  Active: %d, Recently resolved: %d, Combined: %d",
                len(active), len(recently_resolved), len(deduped))

    if len(deduped) < 5:
        logger.error("Too few markets (%d). Exiting.", len(deduped))
        save_run(
            mode="live" if not dry_run else "paper",
            balance_before=balance.balance_usdc if not dry_run else None,
            balance_after=balance.balance_usdc if not dry_run else None,
            markets_scanned=len(deduped), clusters=0,
            relations_discovered=0, trades=[],
        )
        send_summary_notification(
            markets_fetched=len(deduped), relations_discovered=0,
            trades_executed=0, trades_failed=0,
            balance_usdc=balance.balance_usdc if not dry_run else None,
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
        save_run(
            mode="live" if not dry_run else "paper",
            balance_before=balance.balance_usdc if not dry_run else None,
            balance_after=balance.balance_usdc if not dry_run else None,
            markets_scanned=len(deduped), clusters=len(clusters),
            relations_discovered=0, trades=[],
        )
        send_summary_notification(
            markets_fetched=len(deduped), relations_discovered=0,
            trades_executed=0, trades_failed=0,
            balance_usdc=balance.balance_usdc if not dry_run else None,
            dry_run=dry_run,
        )
        return

    # --- Step 3: Check for tradeable signals ---
    logger.info("Step 3: Checking for tradeable signals...")
    active_ids = {m.condition_id for m in active}

    # Filter out markets we're already exposed to
    if not dry_run and SKIP_EXISTING_POSITIONS:
        tradeable_ids = filter_markets_by_exposure(active_ids, exposure)
        skipped = len(active_ids) - len(tradeable_ids)
        if skipped > 0:
            logger.info("  Skipping %d markets with existing positions", skipped)
        active_ids = tradeable_ids

    market_by_question_exact: dict[str, ResolvedMarket] = {m.question: m for m in deduped}
    market_by_question_norm: dict[str, ResolvedMarket] = {}
    for m in deduped:
        market_by_question_norm.setdefault(_normalize_question(m.question), m)
    normalized_market_keys = list(market_by_question_norm.keys())

    trades_executed = 0
    trades_failed = 0
    total_deployed = 0.0
    running_balance = balance.balance_usdc
    trades_remaining = MAX_TRADES_PER_RUN
    trade_records: list[dict] = []

    # Debug counters: helps explain why we may get 0 trades.
    n_relations_failed_market_lookup = 0
    n_attempts_followers_not_active = 0
    n_attempts_leader_not_resolved = 0
    n_attempts_missing_token = 0
    n_attempts_date_mismatch = 0

    for rel in relations:
        if trades_remaining <= 0:
            logger.info("Max trades per run reached (%d), stopping", MAX_TRADES_PER_RUN)
            break

        mi = _resolve_market_by_question(
            rel.question_i,
            deduped,
            market_by_question_exact=market_by_question_exact,
            market_by_question_norm=market_by_question_norm,
            normalized_market_keys=normalized_market_keys,
        )
        mj = _resolve_market_by_question(
            rel.question_j,
            deduped,
            market_by_question_exact=market_by_question_exact,
            market_by_question_norm=market_by_question_norm,
            normalized_market_keys=normalized_market_keys,
        )
        if not mi or not mj:
            n_relations_failed_market_lookup += 1
            continue

        if _is_date_resolution_mismatch(rel.question_i, rel.question_j):
            n_attempts_date_mismatch += 1
            continue

        for leader_candidate, follower_candidate in [(mi, mj), (mj, mi)]:
            if trades_remaining <= 0:
                break
            if follower_candidate.condition_id not in active_ids:
                n_attempts_followers_not_active += 1
                continue

            resolved_data = _check_market_resolved(
                leader_candidate.condition_id, slug=leader_candidate.market_slug
            )
            if not resolved_data or not resolved_data.get("outcome"):
                n_attempts_leader_not_resolved += 1
                continue

            leader_outcome = resolved_data["outcome"]
            signal = _determine_signal(rel, leader_candidate, leader_outcome, follower_candidate)

            if not signal["token_id"]:
                logger.warning("No token_id for follower, skipping")
                n_attempts_missing_token += 1
                continue

            # Check exposure limits before sizing the trade
            if not dry_run:
                can_trade, reason = exposure.can_trade_market(
                    follower_candidate.condition_id,
                    proposed_size_usdc=1.0,  # minimum check
                )
                if not can_trade:
                    logger.info("  Skipping trade: %s", reason)
                    continue

            # Compute dynamic bet size
            bet_size = compute_trade_size(
                running_balance, signal["confidence"], trades_remaining
            ) if not dry_run else 0.0

            logger.info(
                "SIGNAL: BUY %s on '%s' (conf=%.0f%%, size=$%.2f)",
                signal["side"], signal["follower_question"][:55],
                signal["confidence"] * 100, bet_size,
            )

            # Save signal
            sig_path = SIGNALS_DIR / f"signal_{int(time.time())}.json"
            with open(sig_path, "w") as f:
                json.dump({**signal, "bet_size": bet_size}, f, indent=2)

            # Execute or paper-trade
            execution = TradeExecution(success=False, error="Paper trade")
            if not dry_run and bet_size > 0:
                execution = execute_trade(
                    token_id=signal["token_id"],
                    amount_usdc=bet_size,
                )
                if execution.success:
                    trades_executed += 1
                    total_deployed += bet_size
                    running_balance -= bet_size
                    trades_remaining -= 1
                    # Track this position in exposure
                    exposure.add_position(
                        condition_id=follower_candidate.condition_id,
                        token_id=signal["token_id"],
                        side=signal["side"],
                        size_usdc=bet_size,
                        question=signal["follower_question"],
                    )
                    logger.info("Trade OK: order=%s, balance=$%.2f",
                                execution.order_id, running_balance)
                else:
                    trades_failed += 1
                    logger.error("Trade FAILED: %s", execution.error)

            # Notify
            send_trade_notification(
                side=signal["side"],
                follower_question=signal["follower_question"],
                leader_question=signal["leader_question"],
                leader_outcome=leader_outcome,
                confidence=signal["confidence"],
                rationale=signal["rationale"],
                amount_usdc=bet_size if not dry_run else None,
                balance_after=running_balance if not dry_run else None,
                order_id=execution.order_id,
                error=execution.error if not execution.success and not dry_run else None,
            )

            # Record for history
            trade_records.append(record_trade(
                side=signal["side"],
                follower_question=signal["follower_question"],
                follower_condition_id=signal["follower_condition_id"],
                follower_slug=follower_candidate.market_slug,
                token_id=signal["token_id"],
                leader_question=signal["leader_question"],
                leader_outcome=leader_outcome,
                confidence=signal["confidence"],
                rationale=signal["rationale"],
                bet_size_usdc=bet_size,
                order_id=execution.order_id,
                executed=execution.success,
                error=execution.error if not execution.success else None,
            ))

            time.sleep(1)

    # --- Step 4: Save history + Summary ---
    logger.info(
        "Signal debug (why trades may be 0): "
        "market_lookup_fail=%d, date_mismatch=%d, follower_not_active=%d, "
        "leader_not_resolved=%d, missing_token=%d",
        n_relations_failed_market_lookup,
        n_attempts_date_mismatch,
        n_attempts_followers_not_active,
        n_attempts_leader_not_resolved,
        n_attempts_missing_token,
    )
    save_run(
        mode="live" if not dry_run else "paper",
        balance_before=balance.balance_usdc if not dry_run else None,
        balance_after=running_balance if not dry_run else None,
        markets_scanned=len(deduped),
        clusters=len(clusters),
        relations_discovered=len(relations),
        trades=trade_records,
    )

    final_balance = running_balance if not dry_run else None
    logger.info("Pipeline complete. Executed: %d, Failed: %d, Deployed: $%.2f",
                trades_executed, trades_failed, total_deployed)
    send_summary_notification(
        markets_fetched=len(deduped),
        relations_discovered=len(relations),
        trades_executed=trades_executed,
        trades_failed=trades_failed,
        balance_usdc=final_balance,
        total_deployed=total_deployed,
        dry_run=dry_run,
    )


def main():
    parser = argparse.ArgumentParser(description="Semantic Trading — Daily Pipeline")
    parser.add_argument("--max-markets", type=int, default=2000)
    parser.add_argument("--live", action="store_true", help="Force live trading")
    parser.add_argument("--paper", action="store_true", help="Force paper trading")
    args = parser.parse_args()

    if args.paper:
        is_dry_run = True
    elif args.live:
        is_dry_run = False
    else:
        is_dry_run = DRY_RUN

    mode = "LIVE 💰" if not is_dry_run else "PAPER 📝"
    logger.info("Starting Semantic Trading pipeline (%s)", mode)

    try:
        run_pipeline(max_markets=args.max_markets, dry_run=is_dry_run)
    except Exception as e:
        logger.exception("Pipeline crashed")
        send_error_alert(error=e, context="daily pipeline")
        raise


if __name__ == "__main__":
    main()
