from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
POLYMARKET_PRIVATE_KEY: str = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
POLY_API_KEY: str = os.environ.get("POLY_API_KEY", "")
POLY_API_SECRET: str = os.environ.get("POLY_API_SECRET", "")
POLY_PASSPHRASE: str = os.environ.get("POLY_PASSPHRASE", "")
POLY_FUNDER: str = os.environ.get("POLY_FUNDER", "")
DISCORD_WEBHOOK_URL: str = os.environ.get("DISCORD_WEBHOOK_URL", "")
DRY_RUN: bool = os.environ.get("DRY_RUN", "true").lower() in ("true", "1", "yes")

# Position sizing: fraction of available balance to risk per trade.
# With ~93% win rate, full Kelly = 0.86, half-Kelly = 0.43.
# We use ~0.20 per trade with max 4 trades/run = 80% max exposure.
BET_FRACTION: float = float(os.environ.get("BET_FRACTION", "0.20"))
MAX_TRADES_PER_RUN: int = int(os.environ.get("MAX_TRADES_PER_RUN", "4"))
MIN_BET_USDC: float = 2.0
LOW_BALANCE_THRESHOLD: float = 5.0
BANKRUPT_THRESHOLD: float = 1.0

# Exposure management: prevent over-concentration in single markets
# SKIP_EXPOSED_MARKETS: if true, skip any market we already have a position in
# MAX_EXPOSURE_PER_MARKET: max fraction of total balance allowed per market (0.0-1.0)
#   e.g., 0.30 = max 30% of balance in any single market
# When a market is partially exposed, new trades are sized to fill remaining capacity
SKIP_EXPOSED_MARKETS: bool = os.environ.get("SKIP_EXPOSED_MARKETS", "false").lower() in ("true", "1", "yes")
MAX_EXPOSURE_PER_MARKET: float = float(os.environ.get("MAX_EXPOSURE_PER_MARKET", "0.30"))

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

CLUSTER_CATEGORIES = [
    "politics",
    "geopolitics",
    "elections",
    "economy",
    "finance",
    "earnings",
    "crypto",
    "tech",
    "sports",
    "culture",
    "other",
]

CONFIDENCE_THRESHOLD = 0.80
ENTRY_PRICE_CUTOFF = 0.1
TERMINAL_PRICE_CUTOFF = 0.1
MIN_MARKET_DURATION_DAYS = 7
MAX_PAIR_GAP_DAYS = 90
ONLY_SAME_OUTCOME = True

# Sports/game patterns to filter out -- these are independent events with no causal link
SPORTS_EXCLUDE_PATTERNS = [
    "win on 202", "fastest lap", "pole position", "Grand Prix",
    "T20 ", "BPL:", "CSA ", "Toss Match", "Both Teams to Score",
    "end in a draw", "Anytime Goalscorer", "Most kills",
    "Serie A", "Bundesliga", "LaLiga", " vs ",
    "Match O/U", "Completed match",
]
