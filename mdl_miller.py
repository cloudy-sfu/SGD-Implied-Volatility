"""
Corrado-Miller Formula
x = k.*exp(-r*t);
sigma = sqrt(2*pi./t)./(s+x).*(c-0.5*(s-x)+...
    sqrt((c-0.5*(s-x)).^2 - (s-x).^2/pi))
"""
import pickle
from bsm import *
from sklearn.metrics import mean_absolute_error

# %% data
with open("raw/options", "rb") as f:
    s, k, r, c, t = pickle.load(f)
n_samples = s.shape[0]

# %% miller method
x = k * np.exp(-r * t)
sigma = (2 * np.pi / t) ** .5 / (s + x) * (c - .5 * (s - x) +
    np.sqrt((c - .5 * (s - x)) ** 2 - (s - x) ** 2 / np.pi, dtype=np.complex64))
sigma = np.abs(sigma)  # convert complex64 to float32
c_hat = black_scholes_call_value(s, k, t, r, sigma)

# %% save c-hat
print("MAE:", mean_absolute_error(c, c_hat))
with open(f"raw/c-hat/miller", "wb") as f:
    pickle.dump(c_hat, f)
