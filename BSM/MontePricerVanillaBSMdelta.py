import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
import yfinance as yf
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega, delta

# quasi random number Halton
def halton_norm(n, d=1):
    sampler = qmc.Halton(d, scramble=True)
    x_halton = sampler.random(n)
    return stats.norm.ppf(x_halton)


# Input parameters for derivative
S = 105.0 #stock price
K = 100.0  #strike price
vol = 0.1  #volatility as percentage
r = 0.01  #risk-free rate as percentage
Nrsteps = 10  #Nr of time steps/only one is needed but implemented for more complicated SDE`s
Nrsims = 1000  #Nr of simulations
T = ((datetime.date(2023,9,17)-datetime.date(2023,5,17)).days+1)/365 #initial time to expire in years

# Discrete steps
dt = T / Nrsteps
nuxdt = (r - 0.5 * vol ** 2) * dt  # GBM
volxsqdt = vol * np.sqrt(dt)  # GBM
exprxdt = np.exp(r*dt)

# Constant for BSM delta hedge
beta1 = -1

lnS = np.log(S)
sum_C_T = 0
sum_C_T2 = 0


# MC algorithm
for i in range(Nrsims):
    lnS_t = lnS
    c_v = 0
    for j in range(Nrsteps):
        deltaS_t = delta('c', np.exp(lnS_t), K, T - j * dt, r, vol)
        lnS_tn = lnS_t + nuxdt + volxsqdt * np.random.normal()  # GBM
        # z = halton_norm(1)
        # lnS_t = lnS_t + nuxdt + volxsqdt * float(z[0])  # GBM
        c_v = c_v + deltaS_t*(np.exp(lnS_tn) - np.exp(lnS_t)*exprxdt)
        lnS_t = lnS_tn
    # Discount payoff and calc Standard Error
    S_T = np.exp(lnS_t)   # GBM
    C_T = max(0, S_T - K) + beta1*c_v
    sum_C_T = sum_C_T + C_T
    sum_C_T2 = sum_C_T2 + C_T * C_T
# Compute Expectation and SE
C_0 = np.exp(-r * T) * sum_C_T / Nrsims
sigma = np.sqrt((sum_C_T2 - sum_C_T * sum_C_T / Nrsims) * np.exp(-2 * r * T) / (Nrsims - 1))
SE = sigma / np.sqrt(Nrsims)
print("Estimated Price of Call option: ${0} +/- {1}".format(np.round(C_0, 2), np.round(SE, 2)))

# Visualize avg Payoff and SE
x1 = np.linspace(C_0-3*SE, C_0-1*SE, 100)
x2 = np.linspace(C_0-1*SE, C_0+1*SE, 100)
x3 = np.linspace(C_0+1*SE, C_0+3*SE, 100)
s1 = stats.norm.pdf(x1, C_0, SE)
s2 = stats.norm.pdf(x2, C_0, SE)
s3 = stats.norm.pdf(x3, C_0, SE)
plt.fill_between(x1, s1, color='tab:blue',label='> StDev')
plt.fill_between(x2, s2, color='cornflowerblue',label='1 StDev')
plt.fill_between(x3, s3, color='tab:blue')
plt.plot([C_0,C_0],[0, max(s2)*1.1], 'k',
        label='Theoretical Value')
plt.ylabel("Probability")
plt.xlabel("Option Price")
plt.legend()
plt.show()