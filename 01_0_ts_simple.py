# %% ENV
import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()


# %% DESCRIPTION
print("---Description---")
print("Time Series Analysis Simple")
print("Reference:")

print("---Enviroment---")
# %load_ext version_information
%reload_ext version_information
%version_information pandas, matplotlib, sklearn, numpy
%version_information fbprophet, statsmodels

# %% SETTING
print("---Setting---")
# need for plot
%matplotlib inline
# Hyp - the data are inside a sulfolder of data
data_subfolder = 'bitcoin'
# set working directory
wd = os.path.abspath(os.path.dirname("__file__"))
print("The working directory is\n%s" % wd)
# set data directory
data_directory = os.path.abspath(os.path.join(wd, 'data', data_subfolder))
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
df.index = df.index.set_levels([
    df.index.levels[0],
    pd.to_datetime(df.index.levels[1])])

# %% DATASET
print('---Dataset---')
print('-> General Info')
print(df.info())
print("=============================================================")
print(df.describe())
print("=============================================================")
print(df.dtypes)


# %% TIME SERIES
print('---Time Series---')
ts_col = 'Close'


def ts_col2Ts(df, ts_index_level, ts_col):
    # take a column as value
    # and create a time series
    y = pd.Series(
        data=df[ts_col].values,
        name=ts_col,
        index=df.index.levels[ts_index_level])
    y.sort_index(inplace=True)
    return y


series_ts = ts_col2Ts(df, 1, ts_col)
# Basic plot
series_ts.plot()

# 1. load a table csv to df
# 2. take a column as time series df to series
# 3. create forecast period new series
# 4. populate forecast period with forecast

# prophet need an df with the main time series named as 'y'
df_ts = series_ts.to_frame()
df_ts.rename(columns={ts_col: 'y'}, inplace=True)
df_ts['ds'] = df_ts.index

df_ts['cap'] = 10000
m = Prophet(growth='logistic')

m.fit(df_ts)

future = m.make_future_dataframe(periods=60)
future['cap'] = 10000
forecast = m.predict(future)

m.plot(forecast, uncertainty=True)

m.plot_components(forecast)
