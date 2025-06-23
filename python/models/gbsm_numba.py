import numpy as np
from numba import njit, prange
import math

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
