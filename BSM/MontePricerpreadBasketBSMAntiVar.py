import math
import numpy as np
import datetime
import scipy.stats as stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# quasi random number Halton
def halton_norm(n, d=1):
    sampler = qmc.Halton(d, scramble=True)
    x_halton = sampler.random(n)
    return stats.norm.ppf(x_halton)


# Input parameters for derivative
S1 = 101.15  # 30        #stock price
S2 = 101.15  # 30        #stock price
S3 = 101.15  # 30        #stock price
K = (S1-S2)+(S1-S3)+(S2-S3)  #current spread
vol1 = 0.1  # 0.3        #volatility as percentage
vol2 = 0.1  # 0.3        #volatility as percentage
vol3 = 0.1  # 0.3        #volatility as percentage
rho12 = 0.922323 # correlation S1 and S2
rho13 = 0.922323 # correlation S1 and S3
rho23 = 0.922323 # correlation S2 and S3

#correlation matrix
rho = np.array([[1.0,rho12,rho13],
                [rho12,1.0,rho23],
                [rho13,rho23,1.0]])

r = 0.01            #risk-free rate as percentage
Nrsteps = 10              #Nr of time steps/only one is needed but implemented for more complicated SDE`s
Nrsims = 10000            #Nr of simulations
T = ((datetime.date(2022,3,17)-datetime.date(2022,1,17)).days+1)/365  #time in years

# Discrete steps
dt = T / Nrsteps
nu1xdt = (r - 0.5 * vol1 ** 2) * dt  # GBM
vol1xsqdt = vol1 * np.sqrt(dt)  # GBM
lnS_1 = np.log(S1)
nu2xdt = (r - 0.5 * vol2 ** 2) * dt  # GBM
vol2xsqdt = vol2 * np.sqrt(dt)  # GBM
lnS_2 = np.log(S2)
nu3xdt = (r - 0.5 * vol3 ** 2) * dt  # GBM
vol3xsqdt = vol3 * np.sqrt(dt)  # GBM
lnS_3 = np.log(S3)

# (lower) cholesky decomposition
lower_chol = cholesky(rho, lower=True)

sum_C_T = 0
sum_C_T2 = 0



# MC algorithm
for i in range(Nrsims):
    lnS_t1 = lnS_1
    lnS_t2 = lnS_2
    lnS_t3 = lnS_3

    lnS_t1anti = lnS_1
    lnS_t2anti = lnS_2
    lnS_t3anti = lnS_3
    for j in range(Nrsteps):
        # Generate correlated random variables
        Z = np.random.normal(0.0, 1.0, size=(3))
        epsilon = Z @ lower_chol

        lnS_t1 = lnS_t1 + nu1xdt + vol1xsqdt * epsilon[0]  # GBM
        lnS_t2 = lnS_t2 + nu2xdt + vol2xsqdt * epsilon[1]  # GBM
        lnS_t3 = lnS_t3 + nu3xdt + vol3xsqdt * epsilon[2]  # GBM

        lnS_t1anti = lnS_t1anti + nu1xdt - vol1xsqdt * epsilon[0]  # GBM
        lnS_t2anti = lnS_t2anti + nu2xdt - vol2xsqdt * epsilon[1]  # GBM
        lnS_t3anti = lnS_t3anti + nu3xdt - vol3xsqdt * epsilon[2]  # GBM
    # Calc Payoff and average
    S_T1 = np.exp(lnS_t1)  # GBM
    S_T2 = np.exp(lnS_t2)  # GBM
    S_T3 = np.exp(lnS_t3)  # GBM

    S_T1anti = np.exp(lnS_t1anti)  # GBM
    S_T2anti = np.exp(lnS_t2anti)  # GBM
    S_T3anti = np.exp(lnS_t3anti)  # GBM

    C_T = 0.5 * (max(0, (S_T1-S_T2)+(S_T1-S_T3)+(S_T2-S_T3) - K) + max(0, (S_T1anti-S_T2anti)+(S_T1anti-S_T3anti)+(S_T2anti-S_T3anti) - K))
    sum_C_T = sum_C_T + C_T
    sum_C_T2 = sum_C_T2 + C_T * C_T
# Discount payoff and calc Standard Error
C_0 = np.exp(-r * T) * sum_C_T / Nrsims
sigma = np.sqrt((sum_C_T2 - sum_C_T * sum_C_T / Nrsims) * np.exp(-2 * r * T) / (Nrsims - 1))
SE = sigma / np.sqrt(Nrsims)
print("Estimated Price of Call option: ${0} +/- {1}".format(np.round(C_0, 3), np.round(SE, 4)))

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