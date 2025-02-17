import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def sample_mean(returns):
    """Compute the sample mean of returns."""
    return returns.mean(axis=0)

def ewma_mean(returns, lambda_=0.94):
    """Compute the exponentially weighted moving average (EWMA) mean of returns."""
    T, M = returns.shape
    weights = np.array([(1 - lambda_) * lambda_ ** (T - t - 1) for t in range(T)])
    weights /= weights.sum()  # Normalize weights to sum to 1
    return np.dot(weights, returns)

def bayesian_mean(returns, prior_mean=None, prior_precision=0.1):
    """
    Compute the Bayesian estimate of the expected returns.
    Assumes normal prior and normal likelihood.
    """
    sample_mean = returns.mean(axis=0)
    sample_precision = 1.0 / returns.var(axis=0, ddof=1)

    if prior_mean is None:
        prior_mean = np.zeros_like(sample_mean)

    # Bayesian update
    posterior_precision = sample_precision + prior_precision
    posterior_mean = (sample_mean * sample_precision + prior_mean * prior_precision) / posterior_precision

    return posterior_mean

def capm(asset_returns, market_returns, risk_free_rate=0.01):
    """Compute expected returns using the Capital Asset Pricing Model (CAPM)."""
    M = asset_returns.shape[1]
    betas = np.zeros(M)
    expected_returns = np.zeros(M)

    # Compute beta for each asset
    for i in range(M):
        model = LinearRegression()
        model.fit(market_returns.reshape(-1, 1), asset_returns[:, i])
        betas[i] = model.coef_[0]

    # Compute expected returns using CAPM formula: E[R] = Rf + beta * (E[Rm] - Rf)
    market_premium = market_returns.mean() - risk_free_rate
    expected_returns = risk_free_rate + betas * market_premium

    return expected_returns


def returns_apt(asset_returns, macro_factors):
    """Estimate expected returns using Arbitrage Pricing Theory (APT) model."""

    # Find missing dates before reindexing
    missing_dates = set(asset_returns.index) - set(macro_factors.index)
    # print(f"Missing Dates in Macro Factors: {len(missing_dates)}")

    # Align macro factors with asset return dates
    macro_factors = macro_factors.reindex(asset_returns.index).ffill().dropna()

    # Debugging: Print new row counts
    # print(f"After Alignment: Macro Factors: {macro_factors.shape[0]} rows, Asset Returns: {asset_returns.shape[0]} rows")

    if macro_factors.shape[0] != asset_returns.shape[0]:
        raise ValueError(f"Macro factors and asset returns still have mismatched samples: "
                         f"{macro_factors.shape[0]} vs {asset_returns.shape[0]}")

    M = asset_returns.shape[1]  # Number of assets
    K = macro_factors.shape[1]  # Number of macro factors

    betas = np.zeros((M, K))
    expected_returns = np.zeros(M)

    for i in range(M):
        model = LinearRegression()
        model.fit(macro_factors, asset_returns.iloc[:, i])  # Now correctly aligned
        betas[i] = model.coef_

    expected_factor_returns = macro_factors.mean(axis=0)
    expected_returns = np.dot(betas, expected_factor_returns)

    # print("Betas:")
    # print(pd.DataFrame(betas, index=asset_returns.columns, columns=macro_factors.columns))

    return expected_returns


def returns_fama_french(asset_returns, ff_factors):
    """Estimate expected returns using Fama-French 3-Factor Model."""

    # Ensure index alignment: only keep dates present in both datasets
    aligned_data = asset_returns.join(ff_factors, how="inner").dropna()

    # Extract aligned asset returns and factors
    asset_returns_aligned = aligned_data.iloc[:, :asset_returns.shape[1]]
    ff_factors_aligned = aligned_data.iloc[:, asset_returns.shape[1]:]

    M = asset_returns_aligned.shape[1]  # Number of assets
    K = ff_factors_aligned.shape[1]  # Number of factors

    betas = np.zeros((M, K))
    expected_returns = np.zeros(M)

    for i in range(M):
        model = LinearRegression()
        model.fit(ff_factors_aligned, asset_returns_aligned.iloc[:, i])  # Now correctly aligned
        betas[i] = model.coef_

    expected_factor_returns = ff_factors_aligned.mean(axis=0)
    expected_returns = np.dot(betas, expected_factor_returns)


    return expected_returns
