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
DISCORD_WEBHOOK_URL: str = os.environ.get("DISCORD_WEBHOOK_URL", "")
TRADE_SIZE_USDC: float = float(os.environ.get("TRADE_SIZE_USDC", "5.0"))
DRY_RUN: bool = os.environ.get("DRY_RUN", "true").lower() in ("true", "1", "yes")

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
