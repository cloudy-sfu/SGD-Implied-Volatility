import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

with open("raw/options", "rb") as f:
    c = pickle.load(f)[3]

convergence = []
for beginning_point in np.linspace(0.1, 1, 7):
    with open(f"raw/c-hat/newton-20-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        sigma_non_convergence = np.abs(c_hat[:, -1] - c_hat[:, -2]) > 1.e-4
        convergence.append({
            "beginning point": beginning_point,
            "method": "Newton",
            "non-convergence rate": np.mean(sigma_non_convergence),
            "MAE": mean_absolute_error(c[~sigma_non_convergence], c_hat[~sigma_non_convergence, -1])
        })
    with open(f"raw/c-hat/adam-2000-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        sigma_non_convergence = np.abs(c_hat[:, -1] - c_hat[:, -2]) > 1.e-4
        convergence.append({
            "beginning point": beginning_point,
            "method": "GD-Adam",
            "non-convergence rate": np.mean(sigma_non_convergence),
            "MAE": mean_absolute_error(c[~sigma_non_convergence], c_hat[~sigma_non_convergence, -1])
        })
    with open(f"raw/c-hat/adabelief-2000-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        sigma_non_convergence = np.abs(c_hat[:, -1] - c_hat[:, -2]) > 1.e-4
        convergence.append({
            "beginning point": beginning_point,
            "method": "GD-Adabelief",
            "non-convergence rate": np.mean(sigma_non_convergence),
            "MAE": mean_absolute_error(c[~sigma_non_convergence], c_hat[~sigma_non_convergence, -1])
        })
convergence = pd.DataFrame(convergence)
convergence.to_excel("results/compare-convergence.xlsx", index=False)
