"""Polymarket data fetching: Gamma API for metadata, CLOB for price history."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

from semantic_trading.config import (
    CLOB_API_BASE,
    DATA_DIR,
    GAMMA_API_BASE,
    MIN_MARKET_DURATION_DAYS,
    SPORTS_EXCLUDE_PATTERNS,
)
from semantic_trading.types import PricePoint, ResolvedMarket

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Gamma API – market metadata
# ---------------------------------------------------------------------------

def _fetch_gamma_markets(
    *,
    closed: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Fetch a page of markets from the Gamma API."""
    params: dict = {
        "limit": limit,
        "offset": offset,
        "closed": str(closed).lower(),
        "order": "volume",
        "ascending": "false",
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.get(f"{GAMMA_API_BASE}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


def _is_sports_market(question: str) -> bool:
    """Detect sports/game markets that have independent outcomes."""
    q = question.lower()
    return any(pat.lower() in q for pat in SPORTS_EXCLUDE_PATTERNS)


def fetch_resolved_markets(
    *,
    max_markets: int = 1000,
    min_volume: float = 0,
    start_after: Optional[datetime] = None,
    end_before: Optional[datetime] = None,
    month_keyword: Optional[str] = None,
    exclude_sports: bool = True,
) -> list[ResolvedMarket]:
    """
    Fetch resolved binary markets from Polymarket Gamma API.

    Applies the paper's filters:
    - Binary outcomes only (exactly 2 tokens)
    - Duration > MIN_MARKET_DURATION_DAYS
    - Optional: exclude sports/game markets (independent outcomes)
    - Optional month keyword filter (e.g. "April" in question text)
    """
    cache_key = f"resolved_{max_markets}_{month_keyword or 'all'}{'_nosports' if exclude_sports else ''}"
    cache_path = DATA_DIR / f"{cache_key}.json"
    if cache_path.exists():
        logger.info("Loading cached markets from %s", cache_path)
        with open(cache_path) as f:
            raw = json.load(f)
        return [ResolvedMarket(**m) for m in raw]

    markets: list[ResolvedMarket] = []
    offset = 0
    page_size = 100
    max_pages = 100  # safety cap: 10,000 raw markets
    pages_fetched = 0

    while len(markets) < max_markets and pages_fetched < max_pages:
        page = _fetch_gamma_markets(closed=True, limit=page_size, offset=offset)
        if not page:
            break
        pages_fetched += 1
        offset += page_size

        for raw in page:
            try:
                m = _parse_gamma_market(raw)
            except (KeyError, ValueError):
                continue
            if m is None:
                continue

            duration = m.market_end_time - m.market_start_time
            if duration < timedelta(days=MIN_MARKET_DURATION_DAYS):
                continue
            if min_volume and float(raw.get("volumeNum", raw.get("volume", 0))) < min_volume:
                continue
            if start_after and m.market_end_time < start_after:
                continue
            if end_before and m.market_start_time > end_before:
                continue
            if month_keyword and month_keyword.lower() not in m.question.lower():
                continue
            if exclude_sports and _is_sports_market(m.question):
                continue

            markets.append(m)
            if len(markets) >= max_markets:
                break

        time.sleep(0.2)
        if len(markets) >= max_markets:
            break

    logger.info("Scanned %d pages (%d raw markets) -> %d binary markets passed filters",
                pages_fetched, pages_fetched * page_size, len(markets))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump([m.model_dump(mode="json") for m in markets], f, default=str)
    logger.info("Cached %d markets to %s", len(markets), cache_path)

    return markets


def _parse_ts(val: str) -> datetime:
    """Parse an ISO timestamp string into a timezone-aware datetime."""
    if val.endswith("Z"):
        val = val[:-1] + "+00:00"
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_gamma_market(raw: dict, *, require_outcome: bool = True) -> Optional[ResolvedMarket]:
    """Parse a Gamma API market dict into a ResolvedMarket, or None if invalid."""
    question = raw.get("question", "")
    if not question:
        return None

    # Parse outcomes — JSON-encoded string like '["Yes", "No"]'
    outcomes_str = raw.get("outcomes", "")
    try:
        outcomes_list = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(outcomes_list, list) or len(outcomes_list) != 2:
        return None

    # Binary filter: must have exactly Yes/No outcomes
    outcomes_lower = [o.lower() for o in outcomes_list]
    if sorted(outcomes_lower) != ["no", "yes"]:
        return None

    # Determine resolved outcome from outcomePrices
    outcome = ""
    prices_str = raw.get("outcomePrices", "")
    try:
        prices_list = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
    except (json.JSONDecodeError, TypeError):
        prices_list = None

    if prices_list and len(prices_list) == 2:
        try:
            p0, p1 = float(prices_list[0]), float(prices_list[1])
            if p0 > 0.9:
                outcome = outcomes_list[0]
            elif p1 > 0.9:
                outcome = outcomes_list[1]
        except (ValueError, TypeError):
            pass

    if require_outcome and outcome.lower() not in ("yes", "no"):
        return None

    # Parse CLOB token IDs — JSON-encoded string
    token_ids: list[str] = []
    clob_str = raw.get("clobTokenIds", "")
    try:
        token_ids = json.loads(clob_str) if isinstance(clob_str, str) else (clob_str or [])
    except (json.JSONDecodeError, TypeError):
        token_ids = []

    token_list = []
    for i, outcome_label in enumerate(outcomes_list):
        tid = token_ids[i] if i < len(token_ids) else ""
        token_list.append({"token_id": tid, "outcome": outcome_label})

    # Timestamps
    start_date = raw.get("startDate", raw.get("createdAt", "2024-01-01T00:00:00Z"))
    end_date = raw.get("endDate", "2025-01-01T00:00:00Z")

    # Resolution timestamp: prefer umaEndDate, fall back to closedTime
    resolved_on = None
    for ts_field in ("umaEndDate", "closedTime"):
        val = raw.get(ts_field)
        if val:
            try:
                resolved_on = _parse_ts(val)
                break
            except (ValueError, TypeError):
                continue

    # Price proxies for backtesting (CLOB purges history for closed markets)
    last_trade_price = None
    if raw.get("lastTradePrice"):
        try:
            last_trade_price = float(raw["lastTradePrice"])
        except (ValueError, TypeError):
            pass

    best_bid = None
    if raw.get("bestBid"):
        try:
            best_bid = float(raw["bestBid"])
        except (ValueError, TypeError):
            pass

    return ResolvedMarket(
        condition_id=raw.get("conditionId", raw.get("id", "")),
        question=question,
        market_slug=raw.get("slug", ""),
        outcome=outcome.capitalize() if outcome else "",
        market_start_time=_parse_ts(start_date),
        market_end_time=_parse_ts(end_date),
        resolved_on=resolved_on,
        tokens=token_list,
        last_trade_price=last_trade_price,
        best_bid=best_bid,
    )


# ---------------------------------------------------------------------------
# CLOB API – price history
# ---------------------------------------------------------------------------

def fetch_price_history(
    token_id: str,
    *,
    fidelity: int = 60,
) -> list[PricePoint]:
    """
    Fetch price history for a token from the CLOB prices-history endpoint.
    fidelity: time-step in minutes (60 = hourly).
    """
    cache_path = DATA_DIR / "prices" / f"{token_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        return [PricePoint(timestamp=datetime.fromisoformat(p["t"]), price=p["p"]) for p in raw]

    params = {"market": token_id, "interval": "max", "fidelity": fidelity}
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        resp = client.get(f"{CLOB_API_BASE}/prices-history", params=params)
        resp.raise_for_status()
        data = resp.json()

    history = data.get("history", [])
    points: list[PricePoint] = []
    cache_raw: list[dict] = []
    for h in history:
        ts = datetime.fromtimestamp(int(h["t"]), tz=timezone.utc)
        price = float(h["p"])
        points.append(PricePoint(timestamp=ts, price=price))
        cache_raw.append({"t": ts.isoformat(), "p": price})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_raw, f)

    return points


def get_price_at_or_after(
    prices: list[PricePoint],
    target: datetime,
) -> Optional[PricePoint]:
    """Get the first price observation at or after the target timestamp."""
    for p in prices:
        if p.timestamp >= target:
            return p
    return None


def get_terminal_price(prices: list[PricePoint]) -> Optional[float]:
    """Get the last observed price in the series."""
    if not prices:
        return None
    return prices[-1].price


def fetch_active_markets(*, limit: int = 200, exclude_sports: bool = True) -> list[ResolvedMarket]:
    """Fetch currently active (non-resolved) binary markets with pagination."""
    markets: list[ResolvedMarket] = []
    offset = 0
    page_size = 100
    max_pages = 20

    for _ in range(max_pages):
        params = {
            "limit": page_size,
            "offset": offset,
            "closed": "false",
            "order": "volume",
            "ascending": "false",
            "active": "true",
        }
        with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
            resp = client.get(f"{GAMMA_API_BASE}/markets", params=params)
            resp.raise_for_status()
            raw_list = resp.json()

        if not raw_list:
            break

        for raw in raw_list:
            m = _parse_gamma_market(raw, require_outcome=False)
            if m is not None:
                duration = m.market_end_time - m.market_start_time
                if duration >= timedelta(days=MIN_MARKET_DURATION_DAYS):
                    if exclude_sports and _is_sports_market(m.question):
                        continue
                    markets.append(m)

        offset += page_size
        if len(markets) >= limit:
            markets = markets[:limit]
            break
        time.sleep(0.2)

    logger.info("Fetched %d active non-sports markets", len(markets))
    return markets


def fetch_recently_resolved_markets(
    *,
    days_back: int = 3,
    limit: int = 200,
    exclude_sports: bool = True,
) -> list[ResolvedMarket]:
    """Fetch markets that resolved in the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    markets: list[ResolvedMarket] = []

    page = _fetch_gamma_markets(closed=True, limit=limit, offset=0)
    for raw in page:
        m = _parse_gamma_market(raw)
        if m is None:
            continue
        resolved_ts = m.resolved_on or m.market_end_time
        if resolved_ts.tzinfo is None:
            resolved_ts = resolved_ts.replace(tzinfo=timezone.utc)
        if resolved_ts < cutoff:
            continue
        if exclude_sports and _is_sports_market(m.question):
            continue
        markets.append(m)

    logger.info("Fetched %d recently resolved markets (last %d days)", len(markets), days_back)
    return markets
