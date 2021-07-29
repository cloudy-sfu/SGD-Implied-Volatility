import pickle
import pandas as pd
from bsm import *

with open("raw/options", "rb") as f:
    s, k, r, c, t = pickle.load(f)
n_samples = s.shape[0]

x = k * np.exp(-r * t)
sigma = (2 * np.pi / t) ** .5 / (s + x) * (c - .5 * (s - x) +
    np.sqrt((c - .5 * (s - x)) ** 2 - (s - x) ** 2 / np.pi, dtype=np.complex64))
sigma = np.abs(sigma)  # convert complex64 to float32
newton_nonc_sigma_dis = {"sigma (miller)": sigma}
for beginning_point in np.linspace(0.1, 1, 7):
    with open(f"raw/c-hat/newton-20-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
    convergent = ~np.isnan(c_hat[:, -1])
    newton_nonc_sigma_dis[f"newton-{format(beginning_point, '.2f')}"] = convergent.astype('uint8')
newton_nonc_sigma_dis = pd.DataFrame(newton_nonc_sigma_dis)
newton_nonc_sigma_dis.to_excel("results/newton_nonconvergent_sigma_distribution.xlsx", index=False)
