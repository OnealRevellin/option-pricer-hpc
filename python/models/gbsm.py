import numpy as np
import pandas as pd
from scipy.stats import norm
import time

def gbsm_value(
    flavor: str, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    b: float
) -> float:
    """
    Generalized Black Scholes Merton pricing model.

    Parameters
    --------------------
    flavor : string
        Option type ("Call" or "Put").
    S : float or np.ndarray
        Underlying price.
    K : float or np.ndarray
        Strike price.
    T : float or np.ndarray
        Time to maturity (in years).
    r : float or np.ndarray
        Risk-free interest rate (annualized).
    sigma : float or np.ndarray
        Volatility (annualized in %).
    b : float or np.ndarray
        Cost-of-carry.
    --------------------

    Using this model, b can be adapted so the model can match the underlying type.
    For example :
    b = r   ==> Black Scholes model for stock options.
    b = 0.0 ==> Black76 model for options on futures.
    b = r - q ==> Black-Scholes-Merton model for stock options with continuous
        dividend yield.
    """
    sqrtT: float = np.sqrt(T)
    d1: float = (np.log(S / K) + (b + sigma**2 / 2) * T) / (sigma * sqrtT)
    d2: float = d1 - sigma * sqrtT

    call = (
            S * np.exp((b - r) * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )
    
    put = (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * np.exp((b - r) * T) * norm.cdf(-d1)
        )
    
    price = np.where(flavor == 'Call', call, put)
    return price
    
if __name__ == '__main__':
    N = 50_000_000

    flavor = np.random.choice(["Call", "Put"], size=N)
    S = np.random.uniform(50, 150, N).astype(np.float64)
    K = np.random.uniform(50, 150, N).astype(np.float64)
    T = np.random.uniform(0.01, 2.0, N).astype(np.float64)
    r = np.random.uniform(0.0, 0.1, N).astype(np.float64)
    sigma = np.random.uniform(0.1, 0.5, N).astype(np.float64)
    b = np.zeros(N).astype(np.float64)

    start_ns = time.perf_counter_ns()
    prices = gbsm_value(flavor, S, K, T, r, sigma, b)
    end_ns = time.perf_counter_ns()
    exec_time_ns = end_ns - start_ns
    print(f"Execution time: {exec_time_ns} ns")
    print(f"Execution time per option: {exec_time_ns / N} ns")

    print(prices[:10])