import numpy as np
import pandas as pd
import yfinance as yf

from config import START_DATE, END_DATE, TICKERS


def load_rf(start=START_DATE, end=END_DATE) -> pd.Series:
    """3-month T-bill annualised yield (^IRX) → monthly rate, aligned to month-end."""
    raw = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)
    annual = raw["Close"].squeeze() / 100  # ^IRX is quoted as percentage (e.g. 5.25 → 0.0525)
    monthly_annual = annual.resample("ME").last()
    today = pd.Timestamp.today().normalize()
    if monthly_annual.index[-1].year == today.year and monthly_annual.index[-1].month == today.month:
        monthly_annual = monthly_annual.iloc[:-1]
    return (1 + monthly_annual) ** (1 / 12) - 1  # annualised → monthly


def load_prices(tickers=TICKERS, start=START_DATE, end=END_DATE) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    prices = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
    return prices.sort_index().ffill()


def make_monthly(prices: pd.DataFrame):
    monthly_prices = prices.resample("ME").last()
    # Drop the current incomplete month (resample captures partial data as if month-end)
    today = pd.Timestamp.today().normalize()
    if monthly_prices.index[-1].year == today.year and monthly_prices.index[-1].month == today.month:
        monthly_prices = monthly_prices.iloc[:-1]
    monthly_rets = monthly_prices.pct_change()

    daily_vol_ann = prices.pct_change().rolling(60).std() * np.sqrt(252)
    monthly_vol_ann = daily_vol_ann.resample("ME").last().reindex(monthly_rets.index)

    return monthly_prices, monthly_rets, monthly_vol_ann


def load_benchmark(ticker="SPY", start=START_DATE, end=END_DATE) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    monthly = raw["Close"].squeeze().resample("ME").last()
    today = pd.Timestamp.today().normalize()
    if monthly.index[-1].year == today.year and monthly.index[-1].month == today.month:
        monthly = monthly.iloc[:-1]
    return monthly.pct_change().rename(ticker)


if __name__ == "__main__":
    prices = load_prices()
    monthly_prices, monthly_rets, monthly_vol_ann = make_monthly(prices)

    print("Daily prices shape :", prices.shape)
    print("Date range         :", prices.index[0].date(), "→", prices.index[-1].date())
    print("Missing fraction   :\n", prices.isna().mean().sort_values(ascending=False))
    print("\nLatest monthly prices:\n", monthly_prices.tail())
