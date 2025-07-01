import cupy as cp
import time

def erf(x):
    """
    Approximate the error function (erf) using a polynomial approximation.
    """
    sign = cp.sign(x)
    x = cp.abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p  = 0.3275911

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * cp.exp(-x * x)

    return sign * y

def norm_cdf(x):
    return 0.5 * (1 + erf(x / cp.sqrt(2)))

def gbsm_cupy_value(
    flavor: cp.ndarray,
    S: cp.ndarray, 
    K: cp.ndarray, 
    T: cp.ndarray, 
    r: cp.ndarray, 
    sigma: cp.ndarray, 
    b: cp.ndarray
) -> cp.ndarray:
    """
    Generalized Black Scholes Merton pricing model implemented with CuPy for GPU acceleration.

    This function leverages CuPy to:
    - Perform all numerical operations on the GPU, massively accelerating computations 
    by exploiting thousands of CUDA cores in parallel.
    - Utilize efficient GPU implementations of math functions and broadcasting for 
    large-scale array inputs.
    - Achieve significant speedups compared to CPU-based NumPy implementations, 
    especially for very large batches of options.

    Using CuPy allows this function to scale efficiently with data size and GPU hardware,
    making it highly suitable for high-performance option pricing in parallel on GPUs.

    Parameters
    --------------------
    flavor : int (1 for calls 0 for puts) as cp.ndarray
        Option type ("Call" or "Put").
    S : float as cp.ndarray
        Underlying price.
    K : float as cp.ndarray
        Strike price.
    T : float as cp.ndarray
        Time to maturity (in years).
    r : float as cp.ndarray
        Risk-free interest rate (annualized).
    sigma : float as cp.ndarray
        Volatility (annualized in %).
    b : float as cp.ndarray
        Cost-of-carry.
    --------------------

    Using this model, b can be adapted so the model can match the underlying type.
    For example :
    b = r   ==> Black Scholes model for stock options.
    b = 0.0 ==> Black76 model for options on futures.
    b = r - q ==> Black-Scholes-Merton model for stock options with continuous
        dividend yield.
    """
    flavor = cp.asarray(flavor)
    S = cp.asarray(S)
    K = cp.asarray(K)
    T = cp.asarray(T)
    r = cp.asarray(r)
    sigma = cp.asarray(sigma)
    b = cp.asarray(b)

    sqrtT = cp.sqrt(T)
    d1 = (cp.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    call = S * cp.exp((b - r) * T) * norm_cdf(d1) - K * cp.exp(-r * T) * norm_cdf(d2)
    put = K * cp.exp(-r * T) * norm_cdf(-d2) - S * cp.exp((b - r) * T) * norm_cdf(-d1)

    price = cp.where(flavor == 1, call, put)
    return price


if __name__ == '__main__':
    N = 50_000_000
    exec_time_tot = 0
    for i in range(100):
        print(f"\rCounter: {i}/100", end="")
        flavors = cp.ones(N)
        flavors[N//2:] = -1

        S = cp.random.uniform(80, 120, size=N).astype(cp.float64)
        K = cp.random.uniform(80, 120, size=N).astype(cp.float64)
        T = cp.random.uniform(0.1, 2.0, size=N).astype(cp.float64)
        r = cp.random.uniform(0.01, 0.10, size=N).astype(cp.float64)
        sigma = cp.random.uniform(0.1, 0.5, size=N).astype(cp.float64)
        b = cp.zeros(N, dtype=cp.float64)

        start_ns = time.perf_counter_ns()
        prices = gbsm_cupy_value(flavors, S, K, T, r, sigma, b)
        end_ns = time.perf_counter_ns()
        exec_time_ns = end_ns - start_ns
        exec_time_tot += exec_time_ns
        # print(f"Execution time: {exec_time_ns} ns")
        # print(f"Execution time per option: {exec_time_ns / N} ns")

        # print(prices[:10])
    print("")
    exec_time_avg = exec_time_tot / i
    exec_time_avg_per_opt = exec_time_avg / N

    print(f"Execution time: {exec_time_avg} ns")
    print(f"Execution time per option: {exec_time_avg_per_opt} ns")