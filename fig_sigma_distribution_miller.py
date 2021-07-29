import pickle

import numpy as np
import pandas as pd

with open("raw/options", "rb") as f:
    s, k, r, c, t = pickle.load(f)
n_samples = s.shape[0]

x = k * np.exp(-r * t)
sigma = (2 * np.pi / t) ** .5 / (s + x) * (c - .5 * (s - x) +
                                           np.sqrt((c - .5 * (s - x)) ** 2 - (s - x) ** 2 / np.pi, dtype=np.complex64))
sigma = np.abs(sigma)  # convert complex64 to float32

iv = pd.DataFrame({"spot price": s, "strike price": k, "risk-free interest rate": r, "price of call option": c,
      "time of maturity": t, "σ_miller": sigma})
iv.to_excel("results/miller.xlsx", index=False)
