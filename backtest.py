import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COST_BPS, TRAIN_END
from data import load_prices, load_rf, load_benchmark, make_monthly
from strategy import compute_equity_signal, compute_macro_signal, compute_weights


def max_drawdown(equity_curve: pd.Series) -> float:
    return (equity_curve / equity_curve.cummax() - 1.0).min()


def perf_stats(r: pd.Series, periods_per_year=12, bench: pd.Series | None = None, rf=0.0) -> pd.Series:
    r = r.dropna()

    if isinstance(rf, pd.Series):
        rf_m = rf.reindex(r.index).ffill().fillna(0.0)
    else:
        rf_m = (1 + rf) ** (1 / periods_per_year) - 1

    excess = r - rf_m
    vol = r.std()

    ann_ret = (1 + r.mean()) ** periods_per_year - 1
    ann_vol = vol * np.sqrt(periods_per_year)
    sharpe = (excess.mean() / vol) * np.sqrt(periods_per_year) if vol != 0 else np.nan

    dd = excess[excess < 0].std()
    sortino = (excess.mean() / dd) * np.sqrt(periods_per_year) if dd != 0 else np.nan

    mdd = max_drawdown((1 + r).cumprod())
    calmar = ann_ret / abs(mdd) if mdd != 0 else np.nan

    win_rate = (r > 0).mean()
    avg_win  = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan
    win_loss = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else np.nan

    skew = r.skew()
    kurt = r.kurt()

    beta = np.nan
    alpha = np.nan
    if bench is not None:
        b = bench.dropna()
        common = r.index.intersection(b.index)
        if len(common) >= 2:
            rr, bb = r.loc[common], b.loc[common]
            var_b = np.var(bb, ddof=1)
            beta = np.cov(rr, bb, ddof=1)[0, 1] / var_b if var_b != 0 else np.nan
            bench_ann = (1 + bb.mean()) ** periods_per_year - 1
            alpha = ann_ret - beta * bench_ann if not np.isnan(beta) else np.nan

    return pd.Series({
        "Annual Return":  ann_ret,
        "Annual Vol":     ann_vol,
        "Sharpe":         sharpe,
        "Sortino":        sortino,
        "Max Drawdown":   mdd,
        "Win Rate":       win_rate,
        "Avg Win":        avg_win,
        "Avg Loss":       avg_loss,
        "Win/Loss Ratio": win_loss,
        "Skewness":       skew,
        "Beta (SPY)":     beta,
        "Alpha (ann)":    alpha,
    })


def plot_equity_curves(series: dict[str, pd.Series], title: str = "Cumulative Performance (TSMOM)", oos_start: str | None = None):
    plt.figure()
    for label, r in series.items():
        plt.plot((1 + r).cumprod(), label=label)
    if oos_start:
        plt.axvline(pd.Timestamp(oos_start), color="black", linestyle="--", linewidth=1, label="OOS start")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_drawdown(net: pd.Series, oos_start: str | None = None):
    eq = (1 + net).cumprod()
    plt.figure()
    plt.plot(eq / eq.cummax() - 1)
    if oos_start:
        plt.axvline(pd.Timestamp(oos_start), color="black", linestyle="--", linewidth=1, label="OOS start")
        plt.legend()
    plt.title("Net Drawdown - Combined")
    plt.show()


def plot_turnover(turnover: pd.Series):
    plt.figure()
    plt.plot(turnover)
    plt.title("Monthly Turnover — Combined")
    plt.show()


def run_backtest(weights: pd.DataFrame, monthly_rets: pd.DataFrame, cost_bps: int | None = None):
    next_rets = monthly_rets.loc[weights.index].shift(-1)
    gross = (weights * next_rets).sum(axis=1).dropna()
    rate = (cost_bps if cost_bps is not None else COST_BPS) / 10_000
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1).loc[gross.index]
    net = gross - rate * turnover
    return gross, net, turnover


if __name__ == "__main__":
    prices = load_prices()
    monthly_prices, monthly_rets, monthly_vol_ann = make_monthly(prices)
    rf = load_rf()
    spy = load_benchmark("SPY")

    print("=== Signal filter log ===")
    combined, eq_w, macro_w = compute_weights(
        compute_equity_signal(monthly_rets, market_rets=spy),
        compute_macro_signal(monthly_rets, monthly_prices),
        monthly_vol_ann,
    )

    _, net, turnover = run_backtest(combined, monthly_rets)
    spy_aligned = spy.shift(-1).reindex(net.index).dropna()
    net = net.loc[spy_aligned.index]

    splits = {
        f"In-sample  (2006–{TRAIN_END[:4]})":          net.loc[:TRAIN_END],
        f"Out-of-sample ({int(TRAIN_END[:4])+1}–now)": net.loc[TRAIN_END:].iloc[1:],
        "Full period":                                  net,
    }

    rows = {}
    for label, s in splits.items():
        bench = spy_aligned.loc[s.index]
        rows[label] = perf_stats(s, bench=bench, rf=rf)

    print(pd.DataFrame(rows).to_string())
    print(f"\nAvg monthly turnover: {turnover.mean():.3f}")
    print(f"\n*** OOS starts {int(TRAIN_END[:4])+1} ***")
