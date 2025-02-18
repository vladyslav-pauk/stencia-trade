import pandas as pd


def support_resistance_strategy(data, strategy_type, entry_threshold, stop_loss, take_profit):
    """Implements a support-resistance trading strategy with correct cumulative return computation."""

    signals = []
    position = None
    entry_price = None  # Track price at entry

    for i in range(1, len(data)):
        close_price = data["Close"].iloc[i]
        resistance = data["Resistance1"].iloc[i - 1]
        support = data["Support1"].iloc[i - 1]

        if strategy_type == "Support-Resistance":
            # ENTRY SIGNAL: Buy if price dips near support
            if close_price < support * (1 + entry_threshold) and position is None:
                position = "Long"
                entry_price = close_price
                signals.append({"Date": data.index[i], "Action": "Buy", "Price": close_price, "Returns": 0})

            # ENTRY SIGNAL: Sell if price approaches resistance
            elif close_price > resistance * (1 - entry_threshold) and position is None:
                position = "Short"
                entry_price = close_price
                signals.append({"Date": data.index[i], "Action": "Sell", "Price": close_price, "Returns": 0})

            # EXIT SIGNAL: Stop-loss or Take-profit
            elif position == "Long" and (
                close_price <= entry_price * (1 - stop_loss) or close_price >= entry_price * (1 + take_profit)
            ):
                position = None
                returns = (close_price / entry_price) - 1
                signals.append({"Date": data.index[i], "Action": "Sell", "Price": close_price, "Returns": returns})

            elif position == "Short" and (
                close_price >= entry_price * (1 + stop_loss) or close_price <= entry_price * (1 - take_profit)
            ):
                position = None
                returns = (entry_price / close_price) - 1
                signals.append({"Date": data.index[i], "Action": "Buy", "Price": close_price, "Returns": returns})

    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)

    # ✅ Ensure DataFrame always has required columns
    if signals_df.empty:
        signals_df = pd.DataFrame(
            columns=["Date", "Action", "Price", "Returns", "Cumulative Returns", "Relative Returns"])
    else:
        # ✅ Compute Cumulative Returns (Relative to Price Movement)
        signals_df["Cumulative Returns"] = signals_df["Price"] * (1 + signals_df["Returns"]).cumprod()
        signals_df["Relative Returns"] = (1 + signals_df["Returns"]).cumprod()

    return signals_df