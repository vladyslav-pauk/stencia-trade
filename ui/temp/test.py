import yfinance as yf

# symbol = "AAPL"
# start_date = "2022-01-01"
# end_date = "2023-01-01"
#
# data = yf.download(symbol, start=start_date, end=end_date)
# if data.empty:
#     print("No data found.")
# else:
#     print(data.head())

from yahooquery import Ticker

query = "AA"
ticker = Ticker(query, asynchronous=True)

# Fetch details for the symbol
summary = ticker.summary_profile.get(query.upper(), {})
name = summary.get("shortName", "Name not available")
print(f"Symbol: {query.upper()}, Name: {name}")