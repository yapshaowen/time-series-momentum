"""
execution.py — Compute TSMOM weights and write them to IBKR/scores.csv.

Step 1: run this file (from the tsmom/ directory)
Step 2: run IBKR/main.py to place orders

    python execution.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import TICKERS, EQUITY_SLEEVE, MACRO_SLEEVE, LOOKBACK, SKIP
from data import load_prices, load_benchmark, make_monthly
from strategy import compute_equity_signal, compute_macro_signal, compute_weights

SCORES_CSV = Path(__file__).parent.parent / "IBKR" / "scores.csv"


def get_logger() -> logging.Logger:
    logger = logging.getLogger("tsmom_execution")
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def compute_current_weights() -> tuple[dict[str, float], dict[str, float]]:
    """
    Run the full TSMOM pipeline.

    Returns
    -------
    target_weights : {symbol: weight} — vol-targeted weights for all tickers.
        Flat positions are 0.0. Used as the score written to scores.csv so
        IBKR/main.py can size positions proportionally.
    mom_scores : {symbol: momentum_score} — raw 12/1 log-return for reference.
    """
    prices = load_prices()
    monthly_prices, monthly_rets, monthly_vol_ann = make_monthly(prices)
    spy = load_benchmark("SPY")

    combined, _, _ = compute_weights(
        compute_equity_signal(monthly_rets, market_rets=spy),
        compute_macro_signal(monthly_rets, monthly_prices),
        monthly_vol_ann,
    )

    latest = combined.iloc[-1]
    target_weights = {sym: 0.0 for sym in TICKERS}
    for sym, w in latest.items():
        if sym in target_weights:
            target_weights[sym] = float(w)

    all_rets = monthly_rets[EQUITY_SLEEVE + MACRO_SLEEVE]
    mom = np.log1p(all_rets).rolling(LOOKBACK).sum().shift(SKIP).iloc[-1]
    mom_scores = {sym: round(float(v), 6) for sym, v in mom.items() if pd.notna(v)}

    return target_weights, mom_scores


def update_scores_csv(target_weights: dict[str, float], date: pd.Timestamp) -> None:
    """Write target weights as scores so IBKR/main.py sizes positions correctly."""
    date_str = date.date().isoformat()
    rows = pd.DataFrame([
        {"date": date_str, "symbol": sym, "score": round(w, 6)}
        for sym, w in target_weights.items()
    ])

    if SCORES_CSV.exists() and SCORES_CSV.stat().st_size > len("date,symbol,score\n"):
        existing = pd.read_csv(SCORES_CSV)
        existing = existing[existing["date"] != date_str]
        out = pd.concat([existing, rows], ignore_index=True)
    else:
        out = rows

    out.sort_values(["date", "symbol"]).to_csv(SCORES_CSV, index=False)


def main() -> None:
    logger = get_logger()
    logger.info("=== TSMOM Execution ===")
    logger.info("Running pipeline...")

    target_weights, mom_scores = compute_current_weights()
    trade_date = pd.Timestamp.today().normalize()
    update_scores_csv(target_weights, trade_date)
    logger.info("Weights written to %s", SCORES_CSV)

    active = {sym: w for sym, w in target_weights.items() if w != 0.0}
    if not active:
        logger.warning("Kill switch active or no valid signals — all weights zero this month.")
    else:
        logger.info("Target weights for %s:", trade_date.date())
        for sym, w in sorted(active.items(), key=lambda x: -abs(x[1])):
            logger.info("  %-6s  %+.4f  (mom score: %+.4f)", sym, w, mom_scores.get(sym, float("nan")))

    logger.info("Done. Run IBKR/main.py to place orders.")


if __name__ == "__main__":
    main()
