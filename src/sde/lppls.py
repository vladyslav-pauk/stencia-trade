import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution


def lppls_fit(segment, prices):
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(prices[segment[0]:segment[1] + 1])

    def lppls_function(params):
        tc, m, omega, A, B, C1, C2 = params
        trend = A + B * (tc - t) ** m
        oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                       C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
        return np.sum((log_prices - (trend + oscillations)) ** 2)

    bounds = [
        (t[-1] + 1, t[-1] + 10),  # tc
        (0.1, 1),  # m
        (0.1, 100),  # omega
        (-10, 10),  # A
        (-10, 10),  # B
        (-10, 10),  # C1
        (-10, 10)   # C2
    ]

    result = differential_evolution(lppls_function, bounds, maxiter=1000, popsize=10, tol=0.01)
    return result.x


def plot_lppls_fit(prices, ax, segment, params):
    t = np.arange(segment[0], segment[1] + 1)
    log_prices = np.log(prices[segment[0]:segment[1] + 1])
    tc, m, omega, A, B, C1, C2 = params
    trend = A + B * (tc - t) ** m
    oscillations = C1 * (tc - t) ** m * np.cos(omega * np.log(tc - t)) + \
                   C2 * (tc - t) ** m * np.sin(omega * np.log(tc - t))
    fitted = trend + oscillations

    ax.plot(t, log_prices, label="Log-Price (Actual)", color="blue")
    ax.plot(t, fitted, label="LPPLS Fit", color="orange", linestyle="--")
    ax.set_title(f"Segment {segment}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Log-Price")
    ax.legend()
    # ax.show()
