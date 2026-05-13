import numpy as np
import pandas as pd

from config import TRAIN_END
from data import load_prices, load_rf, load_benchmark, make_monthly
from strategy import compute_equity_signal, compute_macro_signal, compute_weights
from backtest import run_backtest, perf_stats, plot_equity_curves, plot_drawdown


def run(plot: bool = True):
    prices = load_prices()
    monthly_prices, monthly_rets, monthly_vol_ann = make_monthly(prices)
    rf  = load_rf()
    spy = load_benchmark("SPY")

    combined, eq_w, macro_w = compute_weights(
        compute_equity_signal(monthly_rets, market_rets=spy),
        compute_macro_signal(monthly_rets, monthly_prices),
        monthly_vol_ann,
    )

    _, net_combined, turnover = run_backtest(combined, monthly_rets)
    spy_aligned = spy.reindex(net_combined.index)

    oos_idx = net_combined.index[net_combined.index > TRAIN_END]
    bench_oos = spy_aligned.loc[oos_idx]
    df = pd.DataFrame({
        "Combined": perf_stats(net_combined.loc[oos_idx], bench=bench_oos, rf=rf),
        "SPY B&H":  perf_stats(bench_oos,                 bench=bench_oos, rf=rf),
    })
    print(f"\n  {int(TRAIN_END[:4])+1} - Present")
    print(df.to_string(float_format="{:.3f}".format))
    print(f"\nAvg monthly turnover: {turnover.mean():.3f}")

    if plot:
        plot_equity_curves({
            "Combined": net_combined,
            "SPY B&H":  spy_aligned,
        }, oos_start=TRAIN_END)
        plot_drawdown(net_combined, oos_start=TRAIN_END)

    return combined, eq_w, macro_w, net_combined


if __name__ == "__main__":
    run()
