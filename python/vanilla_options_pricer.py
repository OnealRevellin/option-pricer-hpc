from dataclasses import dataclass
from enum import Enum
import time

import numpy as np
import pandas as pd

from models.gbsm import gbsm_value


class Models(Enum):
    BS = "Black-Scholes"
    BSM = "Black-Scholes-Merton"
    B76 = "Black76"


class Flavor(Enum):
    Call = "Call"
    Put = "Put"


class VanillaOptionPricer:
    def __init__(
        self,
        model: str,
        flavor: str,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0
    ) -> None:
        self.model = model
        self.flavor = flavor
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q

        self.check_and_adapt_inputs()
        self.compute_cost_of_carry()

    def check_and_adapt_inputs(self):
        for inpt in ["S", "K", "T", "r", "sigma", "q"]:
            attr = getattr(self, inpt)
            
        if isinstance(attr, float) or isinstance(attr, int):
            setattr(self, inpt, np.array([attr], dtype=np.float64))
        elif isinstance(attr, np.ndarray):
            pass
        else:
            raise ValueError(f"{inpt} must be a float, int, or np.ndarray of floats.")

        for inpt in ["flavor"]:
            attr = getattr(self, inpt)

            if isinstance(attr, Flavor):
                repeated = np.full_like(self.S, attr, dtype=object)
                setattr(self, inpt, repeated)

            elif isinstance(attr, np.ndarray):
                if attr.shape != self.S.shape:
                    raise ValueError(
                        f"{inpt} must match shape of S ({self.S.shape})"
                    )
            else:
                raise ValueError(f"{inpt} must be a string or np.ndarray of strings.")
            
        for inpt in ["model"]:
            attr = getattr(self, inpt)

            if isinstance(attr, Models):
                repeated = np.full_like(self.S, attr, dtype=object)
                setattr(self, inpt, repeated)

            elif isinstance(attr, np.ndarray):
                if attr.shape != self.S.shape:
                    raise ValueError(
                        f"{inpt} must match shape of S ({self.S.shape})"
                    )
            else:
                raise ValueError(f"{inpt} must be a string or np.ndarray of strings.")

    def compute_cost_of_carry(self):
        conditions = [
            self.model == Models.B76,
            self.model == Models.BS,
            self.model == Models.BSM
        ]

        values = [
            0.0,
            self.r,
            (self.r - self.q)
        ]

        self.b = np.select(conditions, values, default=None)
        self.b = np.atleast_1d(self.b).astype(np.float64)

    def value(self):
        values = np.where(
            self.flavor == Flavor.Call,
            gbsm_value(
                "Call", 
                S=self.S, K=self.K, T=self.T, r=self.r, 
                sigma=self.sigma, b=self.b
            ),
            np.where(
                self.flavor == Flavor.Put,
                gbsm_value(
                    "Put", 
                    S=self.S, K=self.K, T=self.T, r=self.r, 
                    sigma=self.sigma, b=self.b
                ),
                None
            )
        )

        return values


if __name__ == "__main__":
    N = 1_000_000

    np.random.seed(42)
    S = np.random.uniform(50, 150, N)
    K = np.random.uniform(50, 150, N)
    T = np.random.uniform(0.01, 2.0, N)
    r = np.random.uniform(0.0, 0.1, N)
    sigma = np.random.uniform(0.1, 0.5, N)
    q = np.zeros(N)

    model = Models.B76
    flavor = Flavor.Call

    opt = VanillaOptionPricer(
        model=model,
        flavor=flavor,
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        q=q
    )

    start_ns = time.perf_counter_ns()

    values = opt.value()

    end_ns = time.perf_counter_ns()
    exec_time_ns = end_ns - start_ns
    print(f"Execution time: {exec_time_ns} ns")
    print(f"Execution time per option: {exec_time_ns / N} ns")

    print(values)