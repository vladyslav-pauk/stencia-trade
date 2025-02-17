import numpy as np
import requests


# Check support level proximity
def near_support(price, support, threshold=0.05):
    return abs(price - support) / support < threshold

# Check volume decreasing trend
def volume_trend(df, period=3):
    if len(df) < period:
        return False
    volumes = df["Volume"].tail(period).values
    return np.all(np.diff(volumes) < 0)

# Check price drop from high
def price_drop(df, threshold=20):
    recent_high = df["High"].max()
    current_price = df["Close"].iloc[-1]
    return ((recent_high - current_price) / recent_high) * 100 >= threshold

# Check ForexFactory for high-impact news
def has_high_impact_news():
    url = "https://www.forexfactory.com/"
    try:
        response = requests.get(url)
        if "USD" in response.text and "red" in response.text:
            return True
    except requests.RequestException:
        return False
    return False