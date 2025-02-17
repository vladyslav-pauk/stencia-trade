import yfinance as yf
import pandas as pd
import numpy as np
import random

# Step 1: Get a large list of tickers
def get_all_tickers():
    # Example: Fetch S&P 500 tickers
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(sp500_url)
    tickers = table[0]["Symbol"].tolist()
    return tickers

# Step 2: Randomly sample 100 tickers
all_tickers = get_all_tickers()
random.seed(42)  # For reproducibility
selected_tickers = random.sample(all_tickers, 100)

# Step 3: Download data for the selected tickers
data = yf.download(selected_tickers, start="2010-01-01", end="2023-12-31")['Adj Close']

# Check for missing data
data = data.dropna(axis=1, how='all')  # Remove tickers with no data

# Normalize prices for the formation period
def normalize_prices(prices):
    return prices / prices.iloc[0]

formation_period = 252  # 12 months of trading days
trading_period = 126    # 6 months of trading days

# Ensure we have enough data
if len(data) < formation_period + trading_period:
    raise ValueError("Not enough data for the formation and trading periods.")

formation_prices = data.iloc[:formation_period]
# Remove tickers with zero or insufficient data
valid_tickers = formation_prices.columns[formation_prices.notnull().sum() > formation_period]
formation_prices = formation_prices[valid_tickers]

# Debugging: Check valid tickers after filtering
print(f"Valid tickers: {valid_tickers.tolist()}")
trading_prices = data.iloc[formation_period:formation_period + trading_period]

print(formation_prices.isnull().sum())
print(formation_prices.describe())
formation_prices_normalized = formation_prices.apply(normalize_prices, axis=0)

# Example: Random pair selection and trading logic
from itertools import combinations

def calculate_ssd(pair, prices):
    i, j = pair
    return np.mean((prices[i] - prices[j])**2)

def find_top_pairs(prices, top_n=20):
    pairs = list(combinations(prices.columns, 2))
    ssd_values = {}
    for pair in pairs:
        try:
            ssd_values[pair] = calculate_ssd(pair, prices)
        except Exception as e:
            print(f"Error calculating SSD for pair {pair}: {e}")
    sorted_pairs = sorted(ssd_values, key=ssd_values.get)
    return sorted_pairs[:top_n]

top_pairs = find_top_pairs(formation_prices_normalized)
if not top_pairs:
    raise ValueError("No valid pairs found for analysis.")
print(f"Top pairs: {top_pairs}")

# Trading logic
def trade_pairs(pair, prices, mean, std):
    i, j = pair
    spread = prices[i] - prices[j]
    trades = []
    returns = []
    for t in range(len(spread) - 1):  # Avoid out-of-bounds
        if spread.iloc[t] > mean + 2 * std:  # Divergence: Short i, Long j
            trade_return = (prices[j].iloc[t] - prices[j].iloc[t + 1]) / prices[j].iloc[t] - \
                           (prices[i].iloc[t + 1] - prices[i].iloc[t]) / prices[i].iloc[t]
            trades.append(('SHORT', i, 'LONG', j))
            returns.append(trade_return)
        elif spread.iloc[t] < mean - 2 * std:  # Divergence: Long i, Short j
            trade_return = (prices[i].iloc[t + 1] - prices[i].iloc[t]) / prices[i].iloc[t] - \
                           (prices[j].iloc[t] - prices[j].iloc[t + 1]) / prices[j].iloc[t]
            trades.append(('LONG', i, 'SHORT', j))
            returns.append(trade_return)
    return trades, returns

# Execute trades and calculate returns
all_returns = []
for pair in top_pairs:
    mean_spread = np.mean(formation_prices[pair[0]] - formation_prices[pair[1]])
    std_spread = np.std(formation_prices[pair[0]] - formation_prices[pair[1]])
    print(f"Pair: {pair}, Mean Spread: {mean_spread}, Std Spread: {std_spread}")
    if np.isnan(mean_spread) or np.isnan(std_spread) or std_spread == 0:
        print(f"Invalid statistics for pair {pair}. Skipping.")
        continue
    trades, returns = trade_pairs(pair, trading_prices, mean_spread, std_spread)
    all_returns.extend(returns)

if len(all_returns) == 0:
    print("No trades executed. Check the trade criteria or input data.")
else:
    # Performance Metrics
    total_return = np.sum(all_returns)
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    drawdowns = np.maximum.accumulate(all_returns) - all_returns
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    win_rate = np.sum(np.array(all_returns) > 0) / len(all_returns) if len(all_returns) > 0 else 0
    num_trades = len(all_returns)

    # Display Metrics
    print(f"Total Return: {total_return:.4f}")
    print(f"Mean Return: {mean_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Number of Trades: {num_trades}")