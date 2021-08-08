"""
Newton iteration: x_(k+1) = x_k - f(x_k)/f'(x_k)
"""
import pickle
from bsm import *

with open("raw/options", "rb") as f:
    s, k, r, c, t = pickle.load(f)

n_samples = s.shape[0]
n_steps = 20

for init_sigma in np.linspace(0.1, 1, 7):
    sigma = np.zeros((n_samples, n_steps + 1))
    c_hat = np.zeros((n_samples, n_steps))
    sigma[:, 0] = init_sigma
    vega = np.zeros((n_samples, n_steps))

    for i in range(n_steps):
        vega[:, i] = black_scholes_call_value_derivative(s, k, t, r, sigma[:, i])
        sigma[:, i+1] = sigma[:, i] - \
            (black_scholes_call_value(s, k, t, r, sigma[:, i]) - c) / vega[:, i]
        c_hat[:, i] = black_scholes_call_value(s, k, t, r, sigma[:, i+1])

    with open(f"raw/c-hat/newton-{n_steps}-{format(init_sigma, '.2f')}", "wb") as f:
        pickle.dump(c_hat, f)
