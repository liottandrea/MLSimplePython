# %% DESCRIPTION
"""
https://ntguardian.wordpress.com/2016/09/19/introduction-
stock-market-data-python-1/
"""

# %% ENV
import glob
import pandas as pd
# Package and modules for importing data;
import datetime
# Import matplotlib
import matplotlib.pyplot as plt
from CandleSticks import pandas_candlestick_ohlc
import os
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()  # <== that's all it takes :-)

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


# download dataframe
# data = pdr.get_data_yahoo("AAPL", start="2017-01-01", end="2017-04-30")


# %% SETTING
# This line is necessary for the plot to appear in a Jupyter notebook
%matplotlib inline
# Control the default size of figures in this Jupyter notebook
%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots


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

df.index = df.index.levels[1]

# We will look at stock prices over the past year,
# starting at January 1, 2016
start = datetime.datetime(2016, 1, 1)
end = datetime.date.today()

# %% INPUT

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source
# ("yahoo" for Yahoo! Finance), third is the start date,
# fourth is the end date
apple = web.DataReader("AAPL", "google", start, end)

type(apple)

# look at the table
apple.head()

df.sort_index(inplace=True)
# plot the close figures
df["Close"].plot(grid=True)  # Plot the adjusted closing price of AAPL

# %% CANDELSTICKS
mask = (df['date'] > start_date) & (df['date'] <= end_date)
df.loc[mask]

pandas_candlestick_ohlc(
    df[
        (df.index >= "2017-01-01") &
        (df.index <= "2017-12-31")])

# %% 1+ STOCKS
microsoft = web.DataReader("MSFT", "google", start, end)
google = web.DataReader("GOOG", "google", start, end)

# Below I create a DataFrame consisting of the adjusted closing price
# of these stocks, first by making a list of these objects and using
# the join method
stocks = pd.DataFrame({"AAPL": apple["Close"],
                       "MSFT": microsoft["Close"],
                       "GOOG": google["Close"]})

stocks.head()

stocks.plot(grid=True)

stocks.plot(secondary_y=["AAPL", "MSFT"], grid=True)

# %% COMPUTE RETURN
# df.apply(arg) will apply the function arg to each column in df,
# and return a DataFrame with the result
# Recall that lambda x is an anonymous function accepting parameter x;
# in this case, x will be a pandas Series object
stock_return = stocks.apply(lambda x: x / x[0])
stock_return.head()

stock_return.plot(grid=True).axhline(y=1, color="black", lw=2)
