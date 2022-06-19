# SGD Implied Volatility

Adaptive gradient descent methods for computing implied volatility

![](https://img.shields.io/badge/dependencies-python%203.9-blue.svg)
![](https://img.shields.io/badge/dependencies-tensorflow%202.5-green.svg)

## Introduction

In this paper, a new numerical method based on adaptive gradient descent optimizers is provided for computing the implied volatility from the Black-Scholes (B-S) option pricing model. It is shown that the new method is more accurate than the close form approximation. Compared with the Newton-Raphson method, the new method obtains a reliable rate of convergence and tends to be less sensitive to the beginning point.

## Functions

The European call option pricing model on a stock as follows

$$
\begin{split}
 &   s\Phi(d_1) - ke^{-rt}\Phi(d_2) = c \\
 &   d_1 = \frac{\ln{\frac{s}{k}} + (r + \frac{1}{2}\sigma^2)t}{\sigma\sqrt{t}} \\
 &   d_2 = d_1 - \sigma\sqrt{t},
\end{split}
$$

where $\sigma$ is the volatility, $s$ is the corresponding spot price, $k$ is the strike price, $r$ is the risk-free interest rate, $c$ is the price of the call option, $t$ is the time of maturity, and $\Phi(\cdot)$ is the cumulative distribution function of standard normal distribution up to $x$ i.e.

$$
\Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{x}e^{-\frac{\tau^2}{2}}\mathrm{d}\tau.
$$

All parameters except $\sigma$ can be obtained directly from market
data of the B-S equation. This enables a market-based estimation of a call option's future volatility. Implied volatility can be estimated by the inverse use of B-S equation, which infers $\sigma$ from the market price of the call option.

The numerical approximation of implied volatility from B-S formula is to find the root of

$$
g(\sigma) = s\Phi(d_1) - ke^{-rt}\Phi(d_2) - c = 0.
$$

The numerical approximation $g(\sigma)$ can be transformed into solving an optimization problem

$$
\min_\sigma h(\sigma) = (s\Phi(d_1) - ke^{-rt}\Phi(d_2)-c)^2.
$$

We use adaptive gradient descent optimizer to solve this problem.

## Usage

Dataset: `data/CallOptionBSM.xlsx`

Data pre-processing: `data_pre_processing.py`

The following files are for solving $h(\sigma)$ with different methods:

- `mdl_newton.py`:  Newton-Raphson method
- `mdl_miller.py`:  Corrado-Miller formula
- `mdl_adam.py`:  Adaptive gradient descent method with "Adam" optimizer
- `mdl_adabelief.py`:  Adaptive gradient descent method with "Adabelief" optimizer

The following files are for drawing statistical figures:

- `fig_iter_history.py`: show MAE sequence during the iterative process
- `fig_sigma_distribution_miller.py`: show a histogram of σ estimated with Corrado-Miller formula
- `fig_compare_convergence.py`: select some results to compare NC and MAE metrics

All results are saved in `results` folder.

*This program by default keeps all intermediate results and takes about 25GB hard disk space. For industry use, you can modify the script to minimize the consumption of storage under GPLv3 license.*
