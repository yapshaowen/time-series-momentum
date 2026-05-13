import numpy as np
import pandas as pd

from config import (
    LONG_Q, SHORT_MULT, LOOKBACK, SKIP, VOL_TARGET, MAX_GROSS,
    EQUITY_SLEEVE, MACRO_SLEEVE, EQUITY_TOP_N,
)
from data import load_prices, load_rf, make_monthly


def compute_thresholds(mom: pd.DataFrame, lookback: int) -> pd.DataFrame:
    min_lookback = max(24, lookback * 2)
    abs_panel = mom.abs().stack(future_stack=True).dropna()
    records = []
    for t in mom.index:
        obs = abs_panel[abs_panel.index.get_level_values(0) <= t]
        if len(obs) < min_lookback:
            records.append((t, np.nan, np.nan))
        else:
            long_t = obs.quantile(LONG_Q)
            records.append((t, long_t, SHORT_MULT * long_t))
    return pd.DataFrame(records, columns=["date", "long_thresh", "short_thresh"]).set_index("date")


def compute_equity_signal(
    monthly_rets: pd.DataFrame,
    market_rets: pd.Series | None = None,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
    top_n: int = EQUITY_TOP_N,
) -> pd.DataFrame:
    """
    Rank all equity sectors by 12/1 momentum. Long top N, flat the rest.

    Two independent filters applied in order:
    1. Kill switch (market-level): if SPY 12-month log-return is negative, zero all equity.
    2. Score filter (sector-level): of the top-N candidates, drop any with negative momentum.
    """
    eq_rets = monthly_rets[EQUITY_SLEEVE]
    mom = np.log1p(eq_rets).rolling(lookback).sum().shift(skip)

    spy_mom = None
    if market_rets is not None:
        spy_mom = np.log1p(market_rets).rolling(lookback).sum().shift(skip)
        spy_mom = spy_mom.reindex(mom.index)

    signal = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)

    for t in mom.index:
        if spy_mom is not None:
            spv = spy_mom.loc[t]
            if pd.notna(spv) and spv < 0:
                continue

        row = mom.loc[t].dropna()
        if len(row) < top_n:
            continue
        top = row.nlargest(top_n)
        top_positive = top[top > 0]
        if top_positive.empty:
            continue
        signal.loc[t, top_positive.index] = 1.0

    return signal


def compute_macro_signal(
    monthly_rets: pd.DataFrame,
    monthly_prices: pd.DataFrame | None = None,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
) -> pd.DataFrame:
    """Expanding-window threshold long/short/flat for macro sleeve, with trend filter."""
    macro_rets = monthly_rets[MACRO_SLEEVE]
    mom = np.log1p(macro_rets).rolling(lookback).sum().shift(skip)
    thresholds = compute_thresholds(mom, lookback)

    signal = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    for t, row in thresholds.iterrows():
        if pd.isna(row["long_thresh"]):
            continue
        m = mom.loc[t]
        sig_t = pd.Series(0.0, index=m.index)
        sig_t[m >= row["long_thresh"]] = m[m >= row["long_thresh"]]
        sig_t[m <= -row["short_thresh"]] = m[m <= -row["short_thresh"]]
        signal.loc[t] = sig_t

    if monthly_prices is not None:
        macro_prices = monthly_prices[MACRO_SLEEVE]
        sma = macro_prices.rolling(lookback).mean().shift(skip)
        uptrend = macro_prices.shift(skip) > sma
        signal[(signal > 0) & ~uptrend] = 0.0
        signal[(signal < 0) & uptrend] = 0.0

    return signal


def compute_weights(
    equity_signal: pd.DataFrame,
    macro_signal: pd.DataFrame,
    monthly_vol_ann: pd.DataFrame,
    vol_target: float = VOL_TARGET,
    max_gross: float = MAX_GROSS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (combined, equity_weights, macro_weights).
    Equity: long top EQUITY_TOP_N, each position vol-targeted at vol_target/2.
    Macro: sign-based with diagonal vol scaling (portfolio vol ≈ vol_target).
    Single MAX_GROSS cap applied to the combined portfolio.
    """
    common_idx = equity_signal.index.intersection(macro_signal.index)

    # --- Equity sleeve ---
    eq_vol = monthly_vol_ann[EQUITY_SLEEVE].reindex(common_idx)
    eq_sig = equity_signal.reindex(common_idx)
    eq_w = pd.DataFrame(0.0, index=common_idx, columns=EQUITY_SLEEVE)
    active_eq = eq_sig > 0
    eq_w[active_eq] = (vol_target / 2) / eq_vol[active_eq]
    eq_w = eq_w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Macro sleeve ---
    macro_vol = monthly_vol_ann[MACRO_SLEEVE].reindex(common_idx)
    macro_sig = macro_signal.reindex(common_idx)
    direction = np.sign(macro_sig)
    macro_w = direction * (vol_target / macro_vol)
    macro_w = macro_w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    macro_w[macro_sig == 0.0] = 0.0
    n_active_macro = (macro_w.abs() > 0).sum(axis=1)
    diag_scale = n_active_macro.apply(lambda n: 1.0 / np.sqrt(n) if n > 0 else 1.0)
    macro_w = macro_w.mul(diag_scale, axis=0)

    # --- Combine and filter: require at least one active position in either sleeve ---
    combined = pd.concat([eq_w, macro_w], axis=1)
    mask = (combined.abs() > 0).any(axis=1)
    combined = combined.loc[mask]
    eq_w = eq_w.loc[mask]
    macro_w = macro_w.loc[mask]

    # --- Hard gross leverage cap across both sleeves ---
    gross = combined.abs().sum(axis=1)
    cap_scale = (max_gross / gross).clip(upper=1.0)
    combined = combined.mul(cap_scale, axis=0)
    eq_w = eq_w.mul(cap_scale, axis=0)
    macro_w = macro_w.mul(cap_scale, axis=0)

    return combined, eq_w, macro_w


def apply_rebalance_buffer(weights: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Hold existing weight unless position enters/exits or |change| >= threshold."""
    buffered = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
    prev = pd.Series(0.0, index=weights.columns)
    for t in weights.index:
        target = weights.loc[t]
        enters = (prev == 0.0) & (target != 0.0)
        exits  = (prev != 0.0) & (target == 0.0)
        large  = (target - prev).abs() >= threshold
        trade  = enters | exits | large
        actual = prev.copy()
        actual[trade] = target[trade]
        buffered.loc[t] = actual
        prev = actual
    return buffered
