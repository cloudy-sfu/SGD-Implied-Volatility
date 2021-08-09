import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error

# %% select required files
with open("raw/options", "rb") as f:
    c = pickle.load(f)[3]

# %% draw lines
fig1, fig2, fig3 = plt.figure()
for beginning_point in np.linspace(0.1, 1, 7):
    with open(f"raw/c-hat/newton-20-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        score = np.zeros(c_hat.shape[1])
        for i in range(c_hat.shape[1]):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        fig1.plot(score, label=f'Newton ($\sigma_0$ = {format(beginning_point, ".2f")}, '
                               f'recall = {format(rate_of_recall, ".2f")})')
    with open(f"raw/c-hat/adam-2000-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        score = np.zeros(c_hat.shape[1])
        for i in range(c_hat.shape[1]):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        fig2.plot(score, label=f'Adam ($\sigma_0$ = {format(beginning_point, ".2f")}, '
                               f'recall = {format(rate_of_recall, ".2f")})')
    with open(f"raw/c-hat/adabelief-2000-{format(beginning_point, '.2f')}", "rb") as f:
        c_hat = pickle.load(f)
        score = np.zeros(c_hat.shape[1])
        for i in range(c_hat.shape[1]):
            empty_filter = ~np.isnan(c_hat[:, i])
            score[i] = mean_absolute_error(c[empty_filter], c_hat[empty_filter, i])
        rate_of_recall = empty_filter.mean()
        fig3.plot(score, label=f'Adabelief ($\sigma_0$ = {format(beginning_point, ".2f")}, '
                               f'recall = {format(rate_of_recall, ".2f")})')

# %% decorate figures
fig1.legend()
fig1.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig1.savefig("results/newton_acc_compare.eps")
plt.close()

fig2.legend()
fig2.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig2.savefig("results/adam_acc_compare.eps")
fig2.close()

fig1.legend()
fig3.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log')
plt.xlabel('Step')
plt.ylabel(r'MAE of $\hat{c}$')
fig3.savefig("results/adabelief_acc_compare.eps")
fig3.close()
