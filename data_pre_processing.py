import pandas as pd
from bsm import black_scholes_call_value
import pickle

options_raw = pd.read_excel("data/CallOptionBSM.xlsx", sheet_name="calc_main")

r = .02433  # Shanghai Interbank Offered Rate
not_mature_yet = options_raw['t_days'] > 0
have_solution = black_scholes_call_value(
    options_raw['spot_price'], options_raw['strike_price'],
    options_raw['t_days'], r, 0) - options_raw['settle_price'] < 0
options = options_raw[not_mature_yet & have_solution]

s = options['spot_price'].values  # spot price
k = options['strike_price'].values  # strike price
t = options['t_days'].values  # time of maturity
c = options['settle_price'].values  # call option price

with open("raw/options", "wb") as f:
    pickle.dump([s, k, r, c, t], f)
