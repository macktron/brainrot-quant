#!/usr/bin/env python3
"""
Semantic Trading Backtester

Runs the full pipeline from the paper (arXiv:2512.02436):
1. Fetch resolved Polymarket markets
2. Cluster by semantic similarity
3. Label clusters
4. Discover same/different-outcome relationships
5. Execute leader-follower strategy on historical data
6. Report accuracy and ROI

Usage:
    python run_backtest.py                          # all resolved markets
    python run_backtest.py --month April            # filter to "April" in question text
    python run_backtest.py --trials 30 --month May  # 30 trials for May
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from semantic_trading.backtest import print_backtest_report, run_backtest
from semantic_trading.clustering import cluster_markets
from semantic_trading.data import fetch_resolved_markets
from semantic_trading.discovery import discover_all_relations
from semantic_trading.labeling import label_all_clusters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_trial(markets, trial_num: int = 1):
    """Execute one full pipeline trial."""
    logger.info("=== Trial %d: %d markets ===", trial_num, len(markets))

    logger.info("Step 1: Clustering markets...")
    clusters = cluster_markets(markets)
    logger.info("  -> %d clusters", len(clusters))

    logger.info("Step 2: Labeling clusters...")
    labels = label_all_clusters(clusters)
    for cid, cat in labels.items():
        logger.info("  Cluster %d (%d markets): %s", cid, len(clusters[cid]), cat)

    logger.info("Step 3: Discovering relationships...")
    relations = discover_all_relations(clusters, labels)
    logger.info("  -> %d relations discovered", len(relations))

    if not relations:
        logger.warning("No relations discovered, skipping backtest")
        return None

    logger.info("Step 4: Running backtest...")
    report = run_backtest(relations, markets)
    print_backtest_report(report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Semantic Trading Backtester")
    parser.add_argument("--max-markets", type=int, default=500,
                        help="Max markets to fetch (default: 500)")
    parser.add_argument("--month", type=str, default=None,
                        help="Filter by month keyword in question (e.g. 'April', 'May')")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of trials to run (default: 1)")
    args = parser.parse_args()

    logger.info("Fetching resolved markets (max=%d, month=%s)...",
                args.max_markets, args.month)
    markets = fetch_resolved_markets(
        max_markets=args.max_markets,
        month_keyword=args.month,
    )
    logger.info("Fetched %d markets after filtering", len(markets))

    if len(markets) < 5:
        logger.error("Too few markets (%d) to run pipeline. Try increasing --max-markets "
                      "or removing --month filter.", len(markets))
        sys.exit(1)

    all_reports = []
    for trial in range(1, args.trials + 1):
        report = run_single_trial(markets, trial_num=trial)
        if report and report.total_trades > 0:
            all_reports.append(report)

    if len(all_reports) > 1:
        print("\n" + "=" * 70)
        print(f"AGGREGATE RESULTS OVER {len(all_reports)} TRIALS")
        print("=" * 70)
        accuracies = [r.accuracy for r in all_reports]
        rois = [r.roi for r in all_reports]
        trade_counts = [r.total_trades for r in all_reports]

        acc = np.array(accuracies)
        roi = np.array(rois)
        tc = np.array(trade_counts)

        print(f"  Accuracy  — mean: {acc.mean():.1%}, std: {acc.std():.1%}, "
              f"min: {acc.min():.1%}, max: {acc.max():.1%}")
        print(f"  ROI       — mean: {roi.mean():.1%}, std: {roi.std():.1%}, "
              f"min: {roi.min():.1%}, max: {roi.max():.1%}")
        print(f"  Trades    — mean: {tc.mean():.1f}, total: {tc.sum()}")
        print("=" * 70)
        print()


if __name__ == "__main__":
    main()
