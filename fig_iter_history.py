import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
import re
import pickle
import numpy as np
from matplotlib.ticker import MaxNLocator

# %% select required files
results = sorted(os.listdir("raw/c-hat/"))
with open("raw/options", "rb") as f:
    c = pickle.load(f)[3]

# %% newton methods
fig1 = plt.figure()
for result in results:
    if re.match("newton-", result) is not None:
        _, n_iter, init_sigma = result.split('-')
        n_iter, init_sigma = int(n_iter), float(init_sigma)
        with open(f"raw/c-hat/{result}", "rb") as f:
            c_hat = pickle.load(f)
        score = np.zeros(n_iter)
        for i in range(n_iter):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        plt.plot(score, label=rf'Newton ($\sigma_0$ = {format(init_sigma, ".2f")}, '
                              f'recall = {format(rate_of_recall, ".2f")})')
fig1.legend()
fig1.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig1.savefig("results/newton_acc_compare.eps")
plt.close()

# %% Adam
fig2 = plt.figure()
for result in results:
    if re.match("adam-", result) is not None:
        _, n_iter, init_sigma = result.split('-')
        n_iter, init_sigma = int(n_iter), float(init_sigma)
        with open(f"raw/c-hat/{result}", "rb") as f:
            c_hat = pickle.load(f)
        score = np.zeros(n_iter)
        for i in range(n_iter):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        plt.plot(score, label=rf'Adam ($\sigma_0$ = {format(init_sigma, ".2f")}, '
                              f'recall = {format(rate_of_recall, ".2f")})')
fig2.legend()
fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig2.savefig("results/adam_acc_compare.eps")
plt.close()

# %% Adabelief
fig3 = plt.figure()
for result in results:
    if re.match("adabelief-", result) is not None:
        _, n_iter, init_sigma = result.split('-')
        n_iter, init_sigma = int(n_iter), float(init_sigma)
        with open(f"raw/c-hat/{result}", "rb") as f:
            c_hat = pickle.load(f)
        score = np.zeros(n_iter)
        for i in range(n_iter):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        plt.plot(score, label=rf'Adabelief ($\sigma_0$ = {format(init_sigma, ".2f")}, '
                              f'recall = {format(rate_of_recall, ".2f")})')
fig3.legend()
fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig3.savefig("results/adabelief_acc_compare.eps")
plt.close()
