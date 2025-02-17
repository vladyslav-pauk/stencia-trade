import numpy as np

from src.screener.filters import near_support, volume_trend, price_drop, has_high_impact_news
from src.screener.utils import load_config, get_stock_data, get_PE_ratios


CONFIG_FILE = "config.json"


def scan_stocks(tickers, support=True, volume=True, price=True, pe=True, news=True):
    config = load_config(CONFIG_FILE)
    results = []

    for ticker in tickers:
        print(f"Checking {ticker}...")
        df = get_stock_data(ticker)

        if df.empty:
            print(f"Skipping {ticker} (No Data)")
            continue

        support = config["support_levels"].get(ticker, None)
        pe_trailing, pe_forward = get_PE_ratios(ticker)

        conditions = {
            "near_support": near_support(df["Close"].iloc[-1], support, config["support_threshold"]) if support else False,
            "volume_trend": volume_trend(df, config["volume_period"]),
            "price_drop": price_drop(df, config["price_drop_threshold"]),
            "PE_ratio": (pe_forward < pe_trailing) if not np.isnan(pe_forward) and not np.isnan(pe_trailing) else False,
            "no_high_impact_news": not has_high_impact_news() if config["exclude_news_days"] else True
        }

        if all(conditions.values()):
            results.append(ticker)

    print("\nFiltered Stocks:")
    print(results)

if __name__ == "__main__":
    watchlist = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT"]
    scan_stocks(watchlist, support=True, volume=True, price=True, pe=True, news=True)
