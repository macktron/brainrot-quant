"""LLM-generated short explanation for end-of-run Discord summary."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from semantic_trading.config import LLM_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)


def summarize_run_digest(
    *,
    markets_fetched: int,
    relations_discovered: int,
    trades_executed: int,
    trades_failed: int,
    signal_debug: dict[str, int],
    executed_trades_brief: list[str],
    leader_price_entry_enabled: bool,
    dry_run: bool,
) -> str:
    """
    Produce 2–4 sentences for operators: why trades did or did not happen.
    Falls back to a deterministic string if OPENAI_API_KEY is missing.
    """
    if not OPENAI_API_KEY:
        return _fallback_digest(
            relations_discovered=relations_discovered,
            trades_executed=trades_executed,
            signal_debug=signal_debug,
            leader_price_entry_enabled=leader_price_entry_enabled,
        )

    payload: dict[str, Any] = {
        "markets_fetched": markets_fetched,
        "relations_discovered": relations_discovered,
        "trades_executed": trades_executed,
        "trades_failed": trades_failed,
        "signal_debug": signal_debug,
        "executed_trades_brief": executed_trades_brief[:8],
        "leader_price_proxy_enabled": leader_price_entry_enabled,
        "mode": "paper" if dry_run else "live",
    }

    system = (
        "You write ultra-brief trading ops summaries for Discord (max ~500 chars). "
        "Use plain English. No markdown. "
        "Explain why executed trade count matches the run: e.g. many relations but "
        "no trades because leaders were still open and market prices were not extreme enough, "
        "or followers not in the active set, missing token ids, exposure skips, or below-min bet size. "
        "If signal_debug shows high leader_not_resolved, say pairs were found but leaders were "
        "not yet decided (or not priced above the certainty threshold). "
        "If trades_executed > 0, mention that briefly."
    )
    user = f"Run stats JSON:\n{json.dumps(payload, indent=2)}"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text[:900]
    except Exception as e:
        logger.warning("Run digest LLM failed: %s", e)

    return _fallback_digest(
        relations_discovered=relations_discovered,
        trades_executed=trades_executed,
        signal_debug=signal_debug,
        leader_price_entry_enabled=leader_price_entry_enabled,
    )


def _fallback_digest(
    *,
    relations_discovered: int,
    trades_executed: int,
    signal_debug: dict[str, int],
    leader_price_entry_enabled: bool,
) -> str:
    if relations_discovered == 0:
        return "No logical pairs passed discovery this run."
    if trades_executed > 0:
        return f"{trades_executed} trade(s) executed; see Trades taken below."
    lr = signal_debug.get("leader_not_resolved", 0)
    lip = signal_debug.get("leader_implied_open_price", 0)
    hint = (
        " Open leaders can count as decided when YES or NO is >= threshold (see LEADER_PRICE_CERTAINTY_THRESHOLD)."
        if leader_price_entry_enabled
        else " Leader price-proxy is off (set LEADER_PRICE_CERTAINTY_THRESHOLD > 0)."
    )
    return (
        f"Found {relations_discovered} relations but no fills. "
        f"leader_not_resolved≈{lr} (no decided leader yet or prices not extreme enough); "
        f"open-price signals used≈{lip}.{hint}"
    )
