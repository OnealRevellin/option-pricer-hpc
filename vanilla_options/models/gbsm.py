import numpy as np
import pandas as pd
from scipy.stats import norm


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

    if flavor == "Call":
        return (
            S * np.exp((b - r) * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )
    elif flavor == 'Put':
        return (
            K * np.exp(-r * T) * norm.cdf(-d2)
            - S * np.exp((b - r) * T) * norm.cdf(-d1)
        )
    else:
        raise ValueError("Flavor is not recognized. Please input 'Call' or 'Put'")
    
    
