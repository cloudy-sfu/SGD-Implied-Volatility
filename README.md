# SGD Implied Volatility

Adaptive gradient descent methods for computing implied volatility

![](https://img.shields.io/badge/dependencies-python%203.9-blue)
![](https://img.shields.io/badge/dependencies-tensorflow%202.5-green)

## Installation

```
pip install -r requirements
```
> It is a sufficient package list, but not all packages are necessary.

Require free disk space: 25GB     **important!**

(For industry use, this program does not require so much space. Most of the space are used to record intermediate results for some statistical figures.)

## Acknowledge

https://github.com/tensorflow/tensorflow/

## Usage

Original dataset: `data/CallOptionBSM.xlsx`

Run `data_pre_processing.py` to organize `data/CallOptionBSM.xlsx` to a binary dataset.

The following files are for solving $h(\sigma)$ with different methods:

```
mdl_newton.py  # Newton-Raphson method
mdl_miller.py  # Corrado-Miller formula
mdl_adam.py  # Adaptive gradient descent method with 'Adam' optimizer
mdl_adabelief.py  # Adaptive gradient descent method with 'Adabelief' optimizer
```

The following files are for drawing statistical figures:

```
fig_iter_history.py  # show MAE sequence during the iterative process
fig_sigma_distribution_miller.py  # show a histogram of σ estimated with Corrado-Miller formula
fig_compare_convergence.py  # select some results to compare NC and MAE metrics
```

All results are saved in `results` folder.

