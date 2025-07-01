import numpy as np
from numba import njit, prange
import math
import time

@njit
def norm_cdf(x):
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * math.exp(-x * x / 2.0)
    prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    if x > 0:
        return 1 - prob
    else:
        return prob

@njit(parallel=True, fastmath=True)
def gbsm_numba_value(
    flavor: str, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float, 
    b: float
) -> float:
    """
    Generalized Black Scholes Merton pricing model using Numba for acceleration.

    This function is decorated with @njit(parallel=True, fastmath=True) to leverage Numba's
    Just-In-Time (JIT) compilation capabilities, which significantly speeds up execution by:

    - Compiling Python code to optimized machine code, eliminating interpreter overhead.
    - Enabling automatic parallelization of loops across multiple CPU cores (parallel=True),
    allowing concurrent computation for large input arrays.
    - Activating fast math optimizations (fastmath=True) that allow more aggressive
    floating-point arithmetic transformations for improved performance with minimal
    precision loss.

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
    N = len(S)
    output = np.empty(N, dtype=np.float64)
    for i in prange(N):
        sqrtT = np.sqrt(T[i])
        exprT = np.exp(-r[i] * T[i])
        expbrT = np.exp((b[i] - r[i]) * T[i])
        d1 = (np.log(S[i] / K[i]) + (b[i] + sigma[i]**2 / 2) * T[i]) / (sigma[i] * sqrtT)
        d2 = d1 - sigma[i] * sqrtT

        call_value = (
                S[i] * expbrT * norm_cdf(d1)
                - K[i] * exprT * norm_cdf(d2)
            )

        put_value = (
                K[i] * exprT * norm_cdf(-d2)
                - S[i] * expbrT * norm_cdf(-d1)
            )
        
        is_call = (flavor[i] == "Call")
        is_put = (flavor[i] == "Put")

        output[i] = (is_call * call_value + is_put * put_value)
    
    return output


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
    prices = gbsm_numba_value(flavor, S, K, T, r, sigma, b)
    end_ns = time.perf_counter_ns()
    exec_time_ns = end_ns - start_ns
    print(f"Execution time: {exec_time_ns} ns")
    print(f"Execution time per option: {exec_time_ns / N} ns")

    print(prices[:10])