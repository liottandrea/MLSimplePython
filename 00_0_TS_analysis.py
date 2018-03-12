# %% ENV
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#importing packages for the prediction of time-series data
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
%matplotlib inline

# %% DESCRIPTION
print("---Description---")
print("Time Series Analysis")
print("Reference:")

print("---Enviroment---")
# %load_ext version_information
# %reload_ext version_information
# %version_information pandas, matplotlib, seaborn, numpy, scipy, sklearn, numpy

# %% SETTING
print("---Setting---")
# set working directory
wd = os.path.abspath(os.path.dirname("__file__"))
print("The working directory is\n%s" % wd)
# set data directory
data_directory = os.path.abspath(os.path.join(wd, 'data', 'bitcoin'))
print("The data directory is\n%s" % data_directory)
# set files to load
csv_to_load = glob.glob("%s/*.csv" % data_directory)
print("The file to load are\n%s" % '\n'.join(csv_to_load))
# just extract the names to use as keys later
name_to_load = [f[len(data_directory)+1:-len(".csv")] for f in csv_to_load]


# %% INPUT
print('---Input---')
# read files in the folder and concat into one
df = pd.concat(
    [pd.read_csv(f, index_col='Date', header=0) for f in csv_to_load],
    keys=name_to_load)

print("Converting the index as date")
df.index = df.index.set_levels(
    [df.index.levels[0],
    pd.to_datetime(df.index.levels[1])])
df.head(3)


print('-> General Info')
print(df.info())
print ("=============================================================")
print (df.describe())
print ("=============================================================")
print (df.dtypes)

print('we analysis just close price')
df_ts = df[['Close']]

# simplify the index
df_ts.index = df_ts.index.levels[1]

df_ts.sort_index(inplace=True)
print (type(df_ts))
print ("=============================================================")
print (df_ts.head(3))
print ("=============================================================")
print (df_ts.tail(3))

# Basic plot 
df_ts.plot()

# Dickey Fuller Test Function
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    from statsmodels.tsa.stattools import adfuller
    print('Results of Dickey-Fuller Test:')
    print ("==============================================")
    
    dftest = adfuller(timeseries, autolag='AIC')
    
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    print(dfoutput)

ts = df_ts['Close']
test_stationarity(ts)
# The Test Statistics value is Much higher than critical value.
# So we can't reject the Null Hypothesis.
# Hence Statistically (and obviously from the plot) the Time series
# is Non-Stationary.

# Let's plot the 12-Month Moving Rolling Mean & Variance and find Insights
# Rolling Statistics
rolmean = ts.rolling(window=12).mean()
rolvar = ts.rolling(window=12).std()

plt.plot(ts, label='Original')
plt.plot(rolmean, label='Rolling Mean')
plt.plot(rolvar, label='Rolling Standard Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

# Lets do a quick vanila decomposition to see any trend seasonality etc in the ts
decomposition = sm.tsa.seasonal_decompose(ts, model='multiplicative')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()

# Lets Resample the data by Month and analyze again
df_ts_m = df_ts.resample('M').mean()
print (type(df_ts_m))
print (df_ts_m.head(3))

tsm = df_ts_m['Close']
print (type(tsm))

# Stationarity Check
test_stationarity(tsm)

# Lets do a quick vanila decomposition to see any trend seasonality etc in the ts
decomposition = sm.tsa.seasonal_decompose(tsm, model='multiplicative')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition of multiplicative time series')
plt.show()

# lets try to make the "tsm" Stationary

tsmlog = np.log10(tsm)
tsmlog.dropna(inplace=True)

tsmlogdiff = tsmlog.diff(periods=1)
tsmlogdiff.dropna(inplace=True)
# Stationarity Check
test_stationarity(tsmlogdiff)


# Let's plot ACF & PACF graphs to visualize AR & MA components

fig, axes = plt.subplots(1, 2)
fig.set_figwidth(12)
fig.set_figheight(4)
smt.graphics.plot_acf(tsmlogdiff, lags=30, ax=axes[0], alpha=0.5)
smt.graphics.plot_pacf(tsmlogdiff, lags=30, ax=axes[1], alpha=0.5)
plt.tight_layout()


# 1. load a table csv to df
# 2. take a column as time series df to series
# 3. create forecast period new series
# 4. populate forecast period with forecast 

# 1. load a table csv to df


