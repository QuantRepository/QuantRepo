import math
import numpy as np
import datetime
import scipy.stats as stats
from scipy.stats import qmc
import matplotlib.pyplot as plt
from py_vollib.black_scholes.greeks.analytical import vega, delta, gamma

# quasi random number Halton
def halton_norm(n, d=1):
    sampler = qmc.Halton(d, scramble=True)
    x_halton = sampler.random(n)
    return stats.norm.ppf(x_halton)

# own functions to calculate delta and gamma (not used/ not finished)
def delta_calc(r, S, K, T, sigma, type="c"):
    "Calculate delta of an option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        if type == "c":
            delta_calc = stats.norm.cdf(d1, 0, 1)
        elif type == "p":
            delta_calc = -stats.norm.cdf(-d1, 0, 1)
        return delta_calc
    except:
        print("Either c or p")

def gamma_calc(r, S, K, T, sigma):
    "Calculate delta of an option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        gamma_calc = stats.norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
        return gamma_calc
    except:
        print("Error gamma calc")

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
ergamma = np.exp((2*r+vol**2)*dt) - 2*exprxdt + 1

# Constant for BSM delta and gamma hedge
beta1 = -1
beta2 = -0.5

lnS_1 = np.log(S)
lnS_2 = np.log(S)
sum_C_T = 0
sum_C_T2 = 0


# MC algorithm
for i in range(Nrsims):
    lnS_t1 = lnS_1
    lnS_t2 = lnS_2

    c_v11 = 0
    c_v12 = 0

    cv_21 = 0
    cv_22 = 0
    for j in range(Nrsteps):
        deltaS_t1 = delta('c', np.exp(lnS_t1), K, T - j * dt, r, vol)
        deltaS_t2 = delta('c', np.exp(lnS_t2), K, T - j * dt, r, vol)

        gammaS_t1 = gamma('c', np.exp(lnS_t1), K, T - j * dt, r, vol)
        gammaS_t2 = gamma('c', np.exp(lnS_t2), K, T - j * dt, r, vol)

        epsilon = np.random.normal()
        lnS_tn1 = lnS_t1 + nuxdt + volxsqdt * epsilon  # GBM
        lnS_tn2 = lnS_t2 + nuxdt - volxsqdt * epsilon  # GBM
        # z = halton_norm(1)
        # lnSt = lnSt + nuxdt + volxsqdt * float(z[0])  # GBM

        c_v11 = c_v11 + deltaS_t1 * (np.exp(lnS_tn1) - np.exp(lnS_t1) * exprxdt)
        c_v12 = c_v12 + deltaS_t2 * (np.exp(lnS_tn2) - np.exp(lnS_t2) * exprxdt)

        cv_21 = cv_21 + gammaS_t1*((np.exp(lnS_tn1) - np.exp(lnS_t1)) ** 2 - ergamma * np.exp(lnS_t1) ** 2)
        cv_22 = cv_22 + gammaS_t2*((np.exp(lnS_tn2) - np.exp(lnS_t2)) ** 2 - ergamma * np.exp(lnS_t2) ** 2)

        lnS_t1 = lnS_tn1
        lnS_t2 = lnS_tn2
    # Calc Payoff and average
    S_T1 = np.exp(lnS_t1)   # GBM
    S_T2 = np.exp(lnS_t2)  # GBM

    C_T = 0.5 * ( max(0, S_T1 - K) + beta1*c_v11 + beta2*cv_21 + max(0, S_T2 - K) + beta1*c_v12 + beta2*cv_22 )

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