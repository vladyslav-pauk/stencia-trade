import yfinance as yf
import pandas as pd
import requests
from io import StringIO


def fetch_stock_data(symbol):
    """
    Fetch historical stock data for the given symbol.
    """
    try:
        stock_info = yf.Ticker(symbol)
        history = stock_info.history(period="max")
        if history.empty:
            return None, f"No historical data available for {symbol}."

        return history
    except Exception as e:
        return None, f"Error fetching data: {e}"


def download_fama_french_factors():
    """Download the Fama-French 3-Factor data from Kenneth French's website (CSV format)."""
    file_path = "datasets/Yahoo/F-F_Research_Data_Factors_daily.csv"  # Adjust path as needed

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the line where actual column headers (Mkt-RF, SMB, HML, RF) start
    data_start_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line:  # Look for the correct header line
            data_start_idx = i
            break

    # Ensure that data_start_idx was found
    if data_start_idx is None:
        raise ValueError("Could not find column headers (Mkt-RF, SMB, HML, RF) in the file.")

    # Read the CSV file starting from the correct row
    df = pd.read_csv(file_path, skiprows=data_start_idx)

    # Ensure proper column names
    df.rename(columns={df.columns[0]: "date"}, inplace=True)

    # Convert 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df.set_index("date", inplace=True)

    # Convert percentage values to decimal
    df = df.apply(pd.to_numeric, errors="coerce") / 100

    return df


def download_macro_factors():
    """Download macroeconomic factors (used for APT model) from FRED."""
    # macro_factors = ["CPIAUCSL", "UNRATE", "INDPRO", "GDP", "FEDFUNDS", "M2SL", "VIXCLS"]
    macro_factors = ["CPIAUCSL", "UNRATE", "INDPRO", "FEDFUNDS", "M2SL"]
    macro_data = []

    for factor in macro_factors:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={factor}"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), parse_dates=["observation_date"])
            df.rename(columns={"observation_date": "date", factor: factor.lower()}, inplace=True)
            df.set_index("date", inplace=True)
            macro_data.append(df)
        else:
            raise Exception(f"Failed to download macroeconomic data for {factor}")

    macro_df = pd.concat(macro_data, axis=1).dropna()
    macro_df = macro_df.pct_change().dropna()  # Convert to percentage changes

    return macro_df
