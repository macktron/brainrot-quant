"""
Leader-follower backtester implementing the paper's trading strategy (Section 3.3).

Two evaluation modes:
1. Accuracy: Does the LLM correctly predict same/different outcome? (no price data needed)
2. P&L: Estimated returns using available price proxies or live tick data.

The CLOB API purges history for closed markets, so for resolved-market backtests
we use accuracy evaluation + simplified P&L with a stylized entry price assumption.
For live/forward-testing, real tick data is used.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from semantic_trading.config import ENTRY_PRICE_CUTOFF, TERMINAL_PRICE_CUTOFF
from semantic_trading.data import (
    fetch_price_history,
    get_price_at_or_after,
    get_terminal_price,
)
from semantic_trading.types import (
    BacktestReport,
    MarketRelation,
    PricePoint,
    ResolvedMarket,
    TradeResult,
)

logger = logging.getLogger(__name__)


def _normalize_question(q: str) -> str:
    """Strip date suffixes and normalize whitespace for matching."""
    import re
    q = re.sub(r'\s*\(start:.*$', '', q)
    return q.strip().lower()


def _resolve_market_by_question(
    question: str,
    markets: list[ResolvedMarket],
) -> Optional[ResolvedMarket]:
    """Find a market by its question text (exact match, then normalized, then substring)."""
    for m in markets:
        if m.question == question:
            return m
    nq = _normalize_question(question)
    for m in markets:
        if _normalize_question(m.question) == nq:
            return m
    for m in markets:
        mq = _normalize_question(m.question)
        if nq in mq or mq in nq:
            return m
    return None


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _determine_leader_follower(
    market_i: ResolvedMarket,
    market_j: ResolvedMarket,
) -> tuple[ResolvedMarket, ResolvedMarket]:
    """The leader is the market that resolves first."""
    ts_i = _ensure_aware(market_i.resolved_on or market_i.market_end_time)
    ts_j = _ensure_aware(market_j.resolved_on or market_j.market_end_time)
    if ts_i <= ts_j:
        return market_i, market_j
    return market_j, market_i


def evaluate_relation(
    relation: MarketRelation,
    markets: list[ResolvedMarket],
) -> Optional[TradeResult]:
    """
    Evaluate a single relation for accuracy and estimated P&L.

    For resolved markets without tick-level history, we:
    1. Check accuracy: does is_same_outcome match reality?
    2. Estimate entry price from follower's lastTradePrice or use 0.5 default.
    3. Compute P&L under unit-stake binary payoff.
    """
    market_i = _resolve_market_by_question(relation.question_i, markets)
    market_j = _resolve_market_by_question(relation.question_j, markets)

    if market_i is None or market_j is None:
        logger.debug("Could not resolve markets for relation: %s / %s",
                      relation.question_i[:50], relation.question_j[:50])
        return None

    if not market_i.outcome or not market_j.outcome:
        return None

    leader, follower = _determine_leader_follower(market_i, market_j)
    is_same = relation.is_same_outcome

    leader_outcome_yes = leader.outcome.lower() == "yes"
    buy_yes = (leader_outcome_yes and is_same) or (not leader_outcome_yes and not is_same)
    side = "YES" if buy_yes else "NO"

    # Try real price data first (works for active markets via live runner)
    entry_price = None
    token_id = follower.yes_token_id if buy_yes else follower.no_token_id
    if token_id:
        leader_resolve_time = _ensure_aware(leader.resolved_on or leader.market_end_time)
        prices = fetch_price_history(token_id)
        if prices:
            entry_point = get_price_at_or_after(prices, leader_resolve_time)
            if entry_point:
                entry_price = entry_point.price

    # Fallback: use lastTradePrice from Gamma API as proxy
    if entry_price is None:
        ltp = follower.last_trade_price
        if ltp is not None and 0 < ltp < 1:
            entry_price = ltp if buy_yes else (1 - ltp)
        else:
            entry_price = 0.5

    # Entry extremeness filter
    if entry_price < ENTRY_PRICE_CUTOFF or entry_price > (1 - ENTRY_PRICE_CUTOFF):
        logger.debug("Skipping trade: entry price %.3f outside bounds", entry_price)
        return None

    # Determine correctness
    follower_outcome_yes = follower.outcome.lower() == "yes"
    correct = (buy_yes and follower_outcome_yes) or (not buy_yes and not follower_outcome_yes)

    pnl = (1.0 - entry_price) if correct else -entry_price

    return TradeResult(
        leader_question=leader.question,
        follower_question=follower.question,
        is_same_outcome=is_same,
        confidence_score=relation.confidence_score,
        leader_outcome=leader.outcome,
        follower_outcome=follower.outcome,
        side=side,
        entry_price=entry_price,
        pnl=pnl,
        correct=correct,
    )


def run_backtest(
    relations: list[MarketRelation],
    markets: list[ResolvedMarket],
) -> BacktestReport:
    """Run the full backtest over all discovered relations."""
    trades: list[TradeResult] = []
    for rel in relations:
        result = evaluate_relation(rel, markets)
        if result is not None:
            trades.append(result)

    total_trades = len(trades)
    if total_trades == 0:
        return BacktestReport(
            total_trades=0, correct_trades=0, accuracy=0.0,
            total_invested=0.0, total_pnl=0.0, roi=0.0, trades=[],
        )

    correct_trades = sum(1 for t in trades if t.correct)
    total_invested = sum(t.entry_price for t in trades)
    total_pnl = sum(t.pnl for t in trades)

    return BacktestReport(
        total_trades=total_trades,
        correct_trades=correct_trades,
        accuracy=correct_trades / total_trades,
        total_invested=total_invested,
        total_pnl=total_pnl,
        roi=total_pnl / total_invested if total_invested else 0.0,
        trades=trades,
    )


def print_backtest_report(report: BacktestReport) -> None:
    """Pretty-print backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)
    print(f"  Total trades:     {report.total_trades}")
    print(f"  Correct trades:   {report.correct_trades}")
    print(f"  Accuracy:         {report.accuracy:.1%}")
    print(f"  Total invested:   ${report.total_invested:.2f}")
    print(f"  Total PnL:        ${report.total_pnl:.2f}")
    print(f"  ROI:              {report.roi:.1%}")
    print("=" * 70)

    if report.trades:
        pnls = [t.pnl for t in report.trades]
        arr = np.array(pnls)
        print(f"  PnL  mean:   {arr.mean():.4f}")
        print(f"  PnL  std:    {arr.std():.4f}")
        print(f"  PnL  min:    {arr.min():.4f}")
        print(f"  PnL  max:    {arr.max():.4f}")
        print()
        print("  Individual trades:")
        for i, t in enumerate(report.trades, 1):
            mark = "+" if t.correct else "x"
            print(f"    [{mark}] #{i}: {t.side} @ {t.entry_price:.3f} -> PnL {t.pnl:+.3f}")
            print(f"        Leader:   {t.leader_question[:70]}")
            print(f"        Follower: {t.follower_question[:70]}")
    print()
