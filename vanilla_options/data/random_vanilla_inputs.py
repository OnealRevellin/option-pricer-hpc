"""
Generate a csv containing random inputs for vanilla options pricing.
"""

import numpy as np
import pandas as pd

nb_sims = 50_000
S = np.random.normal(loc=100.0, scale=0.17, size=nb_sims)
K = np.random.normal(loc=100.0, scale=0.17, size=nb_sims)
T = np.random.uniform(0.0, 5.0, nb_sims)
r = np.full(nb_sims, 0.04)
sigma = np.random.normal(loc=0.20, scale=0.04, size=nb_sims)
q = np.zeros(nb_sims)

df = pd.DataFrame({
    'S': S,
    'K': K,
    'T': T,
    'r': r,
    "sigma": sigma,
    'q': q
})

df.to_csv("vanilla_options/data/vanilla_inputs.csv", encoding="utf-8", index=False)