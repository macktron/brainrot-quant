
# Semantic Trading

An automated prediction-market trading system that exploits **logically linked markets** on [Polymarket](https://polymarket.com). Based on the paper *Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets* (Qian et al., 2024).

---

## Strategy Logic

### Core Insight

Prediction markets on Polymarket are priced independently, but many markets are **logically linked** by the same underlying real-world event. When one market resolves, it can reveal information that hasn't yet been priced into a related market. This creates a brief arbitrage-like window.

Example:
"Will the Fed cut rates in June?" and "Will mortgage rates drop below 6% by July?" are not identical markets, but if the Fed cuts rates, mortgage rates dropping becomes far more likely. If the first market resolves YES and the second is still trading at 40 cents, that is a mispricing.

Similarly:
*"Will the Fed cut rates in June?"* resolves YES while *"Will mortgage rates drop below 6% by July?"* is still at $0.40. The first outcome makes the second far more likely — that gap is the trade.

---

### The Leader-Follower Framework

The strategy formalizes this as a **leader-follower** relationship:

* **Leader**: A market that resolves *first* and whose outcome contains information about another market.
* **Follower**: A market that is still active and whose fair price should shift based on the leader's resolution.

The trade: when the leader resolves, immediately take a position in the follower in the direction implied by the leader's outcome. The assumption is that the follower has not yet fully absorbed the new information.

Equivalent formulation:

* **Leader**: Resolves first and is informative.
* **Follower**: Still active and should reprice.

Trading rule: act immediately after leader resolution, betting on delayed price adjustment.

---

### How Relationships Are Found

The strategy uses a three-step pipeline:

1. **Embed & Cluster**
   Market questions are embedded into a vector space (OpenAI `text-embedding-3-small`) and grouped via KMeans. This reduces search complexity from O(n²) to within-cluster comparisons.

2. **Label**
   Each cluster is labeled (politics, crypto, economics, sports, etc.) using an LLM. Sports clusters are excluded since outcomes are generally independent.

3. **Discover**
   Within clusters, an LLM (`gpt-4o`) identifies pairs whose outcomes are **logically determined by the same event**, not merely correlated. Each pair gets:

   * Confidence score
   * Direction flag (same or opposite outcome)

---

### Relationship Discovery Pipeline (Restated)

1. **Embed & Cluster** — Vectorize and group markets.
2. **Label** — Assign semantic categories and remove sports.
3. **Discover** — Identify causally linked pairs with confidence and directionality.

---

### What Makes It Work (and What Doesn’t)

Key insight: **most related markets are actually independent**.

Examples of false relationships:

* "Will Trump cut tariffs?" vs "Will Trump cut taxes?" → unrelated policy decisions
* "ETH above $4400 on Aug 18" vs "ETH reach $4800 in August" → structurally different

Critical filters:

* Only trade **same-outcome relationships**
* Require **high confidence (≥80%)**
* Reject **structural mismatches**
* Enforce **temporal proximity (<90 days)**
* Remove **sports, self-matches**

Backtested accuracy: ~93% on selected trades.

---

### Selectivity (Restated)

* Same-outcome only
* ≥80% confidence
* Structural validation (point-in-time vs interval)
* Temporal constraints
* Sports exclusion
* Self-match removal

---

### Position Sizing

Dynamic sizing based on USDC balance:

* ~20% per trade, scaled by confidence
* Max 30% per trade
* Max 4 trades per run
* Minimum $2

Aggressive but designed for compounding under high accuracy.

---

## Architecture & Infrastructure

### Pipeline (Stateless)

```
Fetch Markets → Embed → Cluster → Label → Discover Relations → Check Resolutions → Trade → Notify → Log
```

Runs twice daily (08:00 / 20:00 UTC). No persistent state between runs.

---

### Architecture (Restated)

Each run is a **single-shot stateless process**:

* No database
* No carryover state
* Full recomputation each run

Manual runs operate in paper trading mode.

---

### Components

| Layer         | What                   | How                                        |
| ------------- | ---------------------- | ------------------------------------------ |
| Data          | Market metadata        | Polymarket Gamma API                       |
| Intelligence  | Embeddings + discovery | OpenAI `text-embedding-3-small` + `gpt-4o` |
| Execution     | Orders + balance       | Polymarket CLOB API (`py-clob-client`)     |
| Notifications | Alerts                 | Discord webhooks                           |
| History       | Logs                   | `history/runs_live.jsonl`                  |
| Scheduling    | Automation             | GitHub Actions cron                        |

---

### Components (Restated)

| Layer         | What                       | How                                      |
| ------------- | -------------------------- | ---------------------------------------- |
| Data          | Market metadata            | Polymarket Gamma API (paginated, cached) |
| Intelligence  | Embeddings + relationships | OpenAI models                            |
| Execution     | Trade placement            | CLOB API                                 |
| Notifications | Alerts                     | Discord                                  |
| History       | Logs                       | JSONL files                              |
| Scheduling    | Cron                       | GitHub Actions                           |

---

### Key Files

```
semantic_trading/
  config.py
  data.py
  clustering.py
  labeling.py
  discovery.py
  backtest.py
  execute.py
  notify.py
  history.py
  types.py

run_backtest.py
run_live.py

.github/workflows/daily_trade.yml
history/
```

---

## Constraints & Limitations

### Structural

* **Latency**: Not real-time; edge decays between runs
* **No exit strategy**: Positions held to resolution
* **FOK orders**: May fail in thin markets

---

### Informational

* **LLM errors** in causal reasoning
* **Embedding limitations** affect clustering
* **3-day resolution window** misses older signals

---

### Market Microstructure

* **Adverse selection**: Fast traders may capture edge first
* **Liquidity risk**: Thin books increase slippage
* **Slippage unmodeled** in backtests

---

### Statistical

* **Small sample size**
* **Survivorship bias**
* **Regime dependence**
* **Correlated losses across trades**

---

### Operational

* **API rate limits**
* **GitHub Actions unreliability**
* **Key management risk**

---

### Constraints & Limitations (Earlier Summary)

* Latency from scheduled runs
* Limited market coverage (~2000 markets, 3-day window)
* LLM reliability constraints
* Liquidity issues
* No exit management
* Single-direction trading only

---

## Setup

1. Install:

```bash
pip install -e .
```

2. Configure `.env`:

```
OPENAI_API_KEY=sk-...
POLYMARKET_PRIVATE_KEY=<key>
DISCORD_WEBHOOK_URL=...
```

3. Add secrets to GitHub:

* OPENAI_API_KEY
* POLYMARKET_PRIVATE_KEY
* DISCORD_WEBHOOK_URL

4. Push to `main` to enable scheduled runs.

---

## References

Qian, Y., Wen, Z., Wang, Z., et al. (2024).
*Semantic Trading: Agentic AI for Clustering and Relationship Discovery in Prediction Markets*.
[https://arxiv.org/abs/2512.02436](https://arxiv.org/abs/2512.02436)

---