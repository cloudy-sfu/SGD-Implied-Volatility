import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error

# %% select required files
results = sorted(os.listdir("raw/c-hat/"))
with open("raw/options", "rb") as f:
    c = pickle.load(f)[3]

recall_error_position = []
name = []
for result in results:
    with open(f"raw/c-hat/{result}", "rb") as f:
        c_hat = pickle.load(f)
    if result == "miller":
        name.append("miller")
        empty_filter = ~np.isnan(c_hat)
        rate_of_recall = empty_filter.mean()
        score = mean_absolute_error(c[empty_filter], c_hat[empty_filter])
        recall_error_position.append([score, rate_of_recall])
    else:
        method, _, init_sigma = result.split('-')
        if method == "newton":
            name.append(f"N{init_sigma}")
        elif method == "adam":
            name.append(f"M{init_sigma}")
        elif method == "adabelief":
            name.append(f"B{init_sigma}")
        empty_filter = ~np.isnan(c_hat[:, -1])
        rate_of_recall = empty_filter.mean()
        score = mean_absolute_error(c[empty_filter], c_hat[empty_filter, -1])
        recall_error_position.append([score, rate_of_recall])
recall_error_position = np.array(recall_error_position)

# %%
ax1 = plt.axes([0.2, 0.15, 0.15, 0.5])
ax1.set_xlim(1e-16, 5e-16)
ax1.set_ylim(auto=True)
ax1.set_xscale('log')
ax1.set_xticks([1e-16, 5e-16])
ax1.scatter(recall_error_position[:, 0], recall_error_position[:, 1])

ax2 = plt.axes()  # standard axes
ax2.set_xscale('log')
ax2.set_xlim(1e-4, 1e-1)
ax2.set_ylim(.9998, 1 + 5e-6)
ax2.scatter(recall_error_position[:, 0], recall_error_position[:, 1])
for label, data in zip(name, recall_error_position):
    ax1.annotate(label, (data[0], data[1]))
    ax2.annotate(label, (data[0], data[1]))
ax2.set_ylabel("Recall %")
ax2.set_xlabel(r"MAE of $\hat{c}$")
plt.savefig("results/recall_acc_compare.eps")
plt.close()
