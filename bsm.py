import numpy as np
from scipy.stats import norm


def black_scholes_call_value(s, k, t, r, sigma):
    """
    看涨期权 BSM 定价公式
    :param s: 标的现货价格
    :param k: 行权价格
    :param t: 有效期
    :param r: SHIBOR 利率
    :param sigma: 隐含波动率
    :return: 期权现价
    """
    d1 = (np.log(s / k) + (r + .5 * sigma ** 2) * t) / (sigma * t ** .5)
    d2 = d1 - sigma * t ** .5
    c = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return c


def black_scholes_call_value_derivative(s, k, t, r, sigma):
    """
    Vega is the sensitivity of striking price over volatility.
    :param s: 标的现货价格
    :param k: 行权价格
    :param t: 有效期
    :param r: SHIBOR 利率
    :param sigma: 隐含波动率
    :return: Vega
    """
    d1 = (np.log(s / k) + (r + .5 * sigma ** 2) * t) / (sigma * t ** .5)
    vega = np.exp(-r*t) * s * t ** .5 * norm.pdf(d1)
    return vega
