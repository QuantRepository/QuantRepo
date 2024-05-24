# Brief summary of Cross-Asset Skew strategy

Strategy ([Nick Baltas et al 2019](https://www.researchgate.net/publication/338498916_Cross-Asset_Skew)): Calculate rolling (256 trading days = 1 trading year) skew of assets (here ETFs) in different asset classes (here Commodities, Equities and Fixed Income), rank them based on their skewness within each asset class and assign weights based on the rank (from low to high skewness). Then construct self financed portfolio (1 dollar short in lower ranked assets 1 dollar long in higher ranked assets) and rebalance each month based on the skewness ranking at the end of the previous month.

Backtest portfolio for different start and end dates by comparing the performance of the skew portfolios with the market portfolios of the individual assets and perform OLS regression to see if strategy yields significant alpha or just market beta.

The skew portfolio from different assets are barely correlated (check) thus advantageous to diversivy and combine the individual portfolios into one diverse portfolio (Global Skewness Factor Portfolio).

Global Skewness Factor Portfolio: Scale each asset class skew portfolio to have a full sample volatility of 10% and combine them all on an equal-weight basis.

Definition of Skew:
\begin{equation}
  S= \frac{1}{N} \sum_{i=1}^{N}\frac{(r_i - \mu)^3}{\sigma^3}
\end{equation}
$\sigma$ rolling std

$\mu$ rolling mean

$r_i$ daily returns of individual ETFs

Definition of weight:
\begin{equation}
  w = z(RANK - \frac{M+1}{2})
\end{equation}
$z$ normalization factor

$RANK$ ranking of ETF within asset class

$M$ nummber of ETFs in asset class
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import statsmodels.api as sm
from scipy.stats import linregress

"""Pull data from yahoo finance"""

# import data
def get_data(stocks, start, end):
    stockData = yf.download(stocks, start, end)
    stockData['Ticker'] = stocks
    #stockData = stockData['Close']
    #returns = stockData.pct_change()
    #meanReturns = returns.mean()
    #covMatrix = returns.cov()
    return stockData  #, covMatrix

"""Set start and end date for the Backtest"""

startdate = "2015-01-01"
enddate = "2024-04-01"

"""# Commodities

Select the ETFs in the commodity asset class and calculate the rolling skew for the last day of the month (EOM) for each ETF.

Make sure that data is clean
"""

commodities = ["GLD", "SLV", "GSG", "USO", "PPLT", "UNG", "DBA"]

alldatacommodities  = []
for j in commodities:
  individualdf =  get_data(j,startdate,enddate)
  individualdf
  individualdf = individualdf.drop(columns=['Open','High','Low','Adj Close','Volume'])
  individualdf['pct_change'] = individualdf.Close.pct_change()
  individualdf['ret'] = np.log(individualdf.Close) - np.log(individualdf.Close.shift(1))
  individualdf['rolling_mean'] = individualdf.ret.rolling(256).mean()
  individualdf['rolling_std'] = individualdf.ret.rolling(256).std()
  individualdf['skew_day'] = ((individualdf.ret-individualdf.rolling_mean)/individualdf.rolling_std)**3
  individualdf['rolling_skew'] = individualdf.skew_day.rolling(256).mean()
  individualdf = individualdf.reset_index()
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['Date']
  individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False).max()
  individualdf['EOM'] = groupings.transform(lambda x: x.max())
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: individualdf[individualdf["Date"] == x.max()].rolling_skew)
  individualdf['EOM_rolling_skew_lookback'] = individualdf.EOM_rolling_skew.shift(1)
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew']
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: x.max())
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew_lookback']
  individualdf['EOM_rolling_skew_lookback'] = groupings.transform(lambda x: x.max())

  alldatacommodities.append(individualdf)

alldatacommodities

"""Add all ETF dataframes into one single dataframe"""

alldatacommodities_df = pd.concat(alldatacommodities)
alldatacommodities_df.tail(5)

"""Plot rolling skew of GLD"""

plt.figure(figsize=(8,2))
alldatacommodities_df[alldatacommodities_df["Ticker"] == "GLD"].rolling_skew.plot()
plt.show()

"""Assign a weight to each ETF in the asset class based on their skew"""

commodities = ["GLD", "SLV", "GSG", "USO", "PPLT", "UNG", "DBA"]
alldatacommodities_df['SkewWeightRaw']=alldatacommodities_df.groupby('Date')['EOM_rolling_skew_lookback'].rank(ascending=False)-(len(commodities)+1)/2
alldatacommodities_df['SkewWeight']=alldatacommodities_df['SkewWeightRaw']/np.sum(np.arange(1, alldatacommodities_df['SkewWeightRaw'].max()+0.1, 1))

"""Plot skew weight of GLD"""

plt.figure(figsize=(8,2))
alldatacommodities_df[alldatacommodities_df["Ticker"] == "GLD"].plot(x='Date', y='SkewWeight')
plt.show()

"""Calculate the returns of the skew portfolio and of the market consisting out of the commodity ETFs"""

alldatacommodities_df['WeightxLogret']=(alldatacommodities_df['SkewWeight']*alldatacommodities_df['ret'])
alldatacommodities_df = alldatacommodities_df[alldatacommodities_df["Date"] >= "2018-01-01"]
groupings = alldatacommodities_df.groupby(['Ticker'],group_keys=False)['WeightxLogret'].cumsum()
alldatacommodities_df['ReturnIndividual'] = groupings.transform(lambda x: x)
groupings = alldatacommodities_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
alldatacommodities_df['PortfolioReturn'] = groupings.transform(lambda x: x)
alldatacommodities_df.tail(50)
groupings = alldatacommodities_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
PortfolioReturnsCommodities = groupings.transform(lambda x: x)
PortfolioReturnsCommodities.plot()

groupings = alldatacommodities_df.groupby(['Ticker'],group_keys=False)['ret'].cumsum()
alldatacommodities_df['MarketReturnIndividual'] = groupings.transform(lambda x: x)

groupings = alldatacommodities_df.groupby(['Date'],group_keys=False)['MarketReturnIndividual'].sum()
MarketReturnsCommodities = groupings.transform(lambda x: x)
MarketReturnsCommodities = MarketReturnsCommodities/len(commodities)
MarketReturnsCommodities.plot()

plt.gca().legend(('Skew Portfolio','Market'))
plt.show

"""Can see that skew portfolio outperforms the market during the testing period

Perform OLS regression to determine alpha and market beta (and their respective p values).
"""

MarketReturnsCommodities1 = sm.add_constant(MarketReturnsCommodities)

result = sm.OLS(PortfolioReturnsCommodities, MarketReturnsCommodities1).fit()

# printing the summary table
print(result.summary())
result.params

linregress(MarketReturnsCommodities,PortfolioReturnsCommodities)

"""# Repeat process for Equity ETFs"""

equity = ["SPY", "EWU", "EWJ", "INDA", "EWG", "EWL", "EWP", "EWQ",
                        "VTI", "FXI", "EWZ", "EWY", "EWA", "EWC", "EWG",
                        "EWH", "EWI", "EWN", "EWD", "EWT", "EZA", "EWW", "ENOR", "EDEN", "TUR"]

alldataequity  = []
for j in equity:
  individualdf =  get_data(j,startdate,enddate)
  individualdf
  individualdf = individualdf.drop(columns=['Open','High','Low','Adj Close','Volume'])
  individualdf['pct_change'] = individualdf.Close.pct_change()
  individualdf['ret'] = np.log(individualdf.Close) - np.log(individualdf.Close.shift(1))
  individualdf['rolling_mean'] = individualdf.ret.rolling(256).mean()
  individualdf['rolling_std'] = individualdf.ret.rolling(256).std()
  individualdf['skew_day'] = ((individualdf.ret-individualdf.rolling_mean)/individualdf.rolling_std)**3
  individualdf['rolling_skew'] = individualdf.skew_day.rolling(256).mean()
  individualdf = individualdf.reset_index()
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['Date']
  individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False).max()
  individualdf['EOM'] = groupings.transform(lambda x: x.max())
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: individualdf[individualdf["Date"] == x.max()].rolling_skew)
  individualdf['EOM_rolling_skew_lookback'] = individualdf.EOM_rolling_skew.shift(1)
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew']
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: x.max())
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew_lookback']
  individualdf['EOM_rolling_skew_lookback'] = groupings.transform(lambda x: x.max())

  alldataequity.append(individualdf)

#alldataequity

alldataequity_df = pd.concat(alldataequity)

alldataequity_df['SkewWeightRaw']=alldataequity_df.groupby('Date')['EOM_rolling_skew_lookback'].rank(ascending=False)-(len(equity)+1)/2
alldataequity_df['SkewWeight']=alldataequity_df['SkewWeightRaw']/np.sum(np.arange(1, alldataequity_df['SkewWeightRaw'].max()+0.1, 1))

alldataequity_df['WeightxLogret']=(alldataequity_df['SkewWeight']*alldataequity_df['ret'])
alldataequity_df = alldataequity_df[alldataequity_df["Date"] >= "2018-01-01"]
groupings = alldataequity_df.groupby(['Ticker'],group_keys=False)['WeightxLogret'].cumsum()
alldataequity_df['ReturnIndividual'] = groupings.transform(lambda x: x)
groupings = alldataequity_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
alldataequity_df['PortfolioReturn'] = groupings.transform(lambda x: x)
alldataequity_df.tail(50)
groupings = alldataequity_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
PortfolioReturnsEquity = groupings.transform(lambda x: x)
PortfolioReturnsEquity.plot()

groupings = alldataequity_df.groupby(['Ticker'],group_keys=False)['ret'].cumsum()
alldataequity_df['MarketReturnIndividual'] = groupings.transform(lambda x: x)

groupings = alldataequity_df.groupby(['Date'],group_keys=False)['MarketReturnIndividual'].sum()
MarketReturnsEquity = groupings.transform(lambda x: x)
MarketReturnsEquity = MarketReturnsEquity/len(equity)
MarketReturnsEquity.plot()

plt.gca().legend(('Skew Portfolio','Market'))
plt.show

MarketReturnsEquity1 = sm.add_constant(MarketReturnsEquity)

result = sm.OLS(PortfolioReturnsEquity, MarketReturnsEquity1).fit()

# printing the summary table
print(result.summary())
result.params

linregress(MarketReturnsEquity,PortfolioReturnsEquity)

"""# Repeat process for fixed income ETFs"""

FI = ["AGG", "TLT", "LQD", "JNK", "MUB", "MBB", "IGOV", "EMB", "BND", "BNDX", "VCIT", "VCSH", "BSV", "SRLN"]

alldataFI  = []
for j in FI:
  individualdf =  get_data(j,startdate,enddate)
  individualdf
  individualdf = individualdf.drop(columns=['Open','High','Low','Adj Close','Volume'])
  individualdf['pct_change'] = individualdf.Close.pct_change()
  individualdf['ret'] = np.log(individualdf.Close) - np.log(individualdf.Close.shift(1))
  individualdf['rolling_mean'] = individualdf.ret.rolling(256).mean()
  individualdf['rolling_std'] = individualdf.ret.rolling(256).std()
  individualdf['skew_day'] = ((individualdf.ret-individualdf.rolling_mean)/individualdf.rolling_std)**3
  individualdf['rolling_skew'] = individualdf.skew_day.rolling(256).mean()
  individualdf = individualdf.reset_index()
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['Date']
  individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False).max()
  individualdf['EOM'] = groupings.transform(lambda x: x.max())
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: individualdf[individualdf["Date"] == x.max()].rolling_skew)
  individualdf['EOM_rolling_skew_lookback'] = individualdf.EOM_rolling_skew.shift(1)
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew']
  individualdf['EOM_rolling_skew'] = groupings.transform(lambda x: x.max())
  groupings = individualdf.groupby([individualdf.Date.dt.year, individualdf.Date.dt.month],group_keys=False)['EOM_rolling_skew_lookback']
  individualdf['EOM_rolling_skew_lookback'] = groupings.transform(lambda x: x.max())

  alldataFI.append(individualdf)

#alldataFI

alldataFI_df = pd.concat(alldataFI)

alldataFI_df['SkewWeightRaw']=alldataFI_df.groupby('Date')['EOM_rolling_skew_lookback'].rank(ascending=False)-(len(FI)+1)/2
alldataFI_df['SkewWeight']=alldataFI_df['SkewWeightRaw']/np.sum(np.arange(1, alldataFI_df['SkewWeightRaw'].max()+0.1, 1))

alldataFI_df['WeightxLogret']=(alldataFI_df['SkewWeight']*alldataFI_df['ret'])
alldataFI_df = alldataFI_df[alldataFI_df["Date"] >= "2018-01-01"]
groupings = alldataFI_df.groupby(['Ticker'],group_keys=False)['WeightxLogret'].cumsum()
alldataFI_df['ReturnIndividual'] = groupings.transform(lambda x: x)
groupings = alldataFI_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
alldataFI_df['PortfolioReturn'] = groupings.transform(lambda x: x)
alldataFI_df.tail(50)
groupings = alldataFI_df.groupby(['Date'],group_keys=False)['ReturnIndividual'].sum()
PortfolioReturnsFI = groupings.transform(lambda x: x)
PortfolioReturnsFI.plot()

groupings = alldataFI_df.groupby(['Ticker'],group_keys=False)['ret'].cumsum()
alldataFI_df['MarketReturnIndividual'] = groupings.transform(lambda x: x)

groupings = alldataFI_df.groupby(['Date'],group_keys=False)['MarketReturnIndividual'].sum()
MarketReturnsFI = groupings.transform(lambda x: x)
MarketReturnsFI = MarketReturnsFI/len(FI)
MarketReturnsFI.plot()

plt.gca().legend(('Skew Portfolio','Market'))
plt.show

MarketReturnsFI1 = sm.add_constant(MarketReturnsFI)

result = sm.OLS(PortfolioReturnsFI, MarketReturnsFI1).fit()

# printing the summary table
print(result.summary())
result.params

linregress(MarketReturnsFI,PortfolioReturnsFI)

"""# Global Skewness Factor Portfolio

Skew portfolios from different assets have low correlation (check) => combine them into one diverse portfolio.

Scale each asset class skew portfolio to have a full sample
volatility of 10% and combine them all on an equal-weight basis.
"""

ReturnsFI_df = PortfolioReturnsFI.to_frame()
ReturnsFI_df["MarketReturnIndividual"] = MarketReturnsFI
ReturnsFI_df["Asset"] = "FI"

ReturnsEquity_df = PortfolioReturnsEquity.to_frame()
ReturnsEquity_df["MarketReturnIndividual"] = MarketReturnsEquity
ReturnsEquity_df["Asset"] = "Equity"

ReturnsCommodities_df = PortfolioReturnsCommodities.to_frame()
ReturnsCommodities_df["MarketReturnIndividual"] = MarketReturnsCommodities
ReturnsCommodities_df["Asset"] = "Commodities"

ReturnsCombined_df = ReturnsCommodities_df._append(ReturnsEquity_df, ignore_index=False)
ReturnsCombined_df = ReturnsCombined_df._append(ReturnsFI_df, ignore_index=False)

ReturnsCombined_df = ReturnsCombined_df.reset_index()
ReturnsCombined_df


groupings = ReturnsCombined_df.groupby(['Asset'],group_keys=False)['ReturnIndividual']
ReturnsCombined_df['Std'] = groupings.transform(lambda x: x.rolling(256).std())
ReturnsCombined_df['NormReturnIndividual'] = 0.1*ReturnsCombined_df.ReturnIndividual/ReturnsCombined_df.Std
ReturnsCombined_df['NormMarketReturnIndividual'] = 0.1*ReturnsCombined_df.MarketReturnIndividual/ReturnsCombined_df.Std
ReturnsCombined_df

groupings = ReturnsCombined_df.groupby(['Date'],group_keys=False)['NormReturnIndividual'].mean()
GSF_Portfolio = groupings.transform(lambda x: x)
groupings = ReturnsCombined_df.groupby(['Date'],group_keys=False)['NormMarketReturnIndividual'].mean()
GSF_Market = groupings.transform(lambda x: x)
GSF_Portfolio.plot()
GSF_Market.plot()

plt.gca().legend(('GSF Portfolio','Market'))
plt.show

GSF_Market.dropna()
GSF_Market.dropna()

GSF_Market1 = sm.add_constant(GSF_Market)

result = sm.OLS(GSF_Portfolio.dropna(), GSF_Market1.dropna()).fit()

# printing the summary table
print(result.summary())
result.params

"""TODO: Check if skew is providing new information that is not available from Value, Momentum, and Carry Factors

For equity ETFs simple to check via value/momentum/carry ETFs
"""
