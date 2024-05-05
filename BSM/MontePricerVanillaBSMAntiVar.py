import numpy as np
import datetime
import scipy.stats as stats
from scipy.stats import qmc
import matplotlib.pyplot as plt

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


lnS = np.log(S)
sum_C_T = 0
sum_C_T2 = 0


# MC algorithm
for i in range(Nrsims):
    lnS1_t = lnS
    lnS2_t = lnS
    for j in range(Nrsteps):
        epsilon = np.random.normal()
        lnS1_t = lnS1_t + nuxdt + volxsqdt * epsilon  # GBM
        lnS2_t = lnS2_t + nuxdt - volxsqdt * epsilon  # GBM
    # Calc Payoff and average
    S1_T = np.exp(lnS1_t)  # GBM
    S2_T = np.exp(lnS2_t)  # GBM
    CT = 0.5 * ( max(0, S1_T - K) + max(0, S2_T - K) )
    sum_C_T = sum_C_T + CT
    sum_C_T2 = sum_C_T2 + CT * CT
# Discount payoff and calc Standard Error
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