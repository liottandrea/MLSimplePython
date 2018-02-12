#%% DESCRIPTION
"""
https://ntguardian.wordpress.com/2016/09/19/introduction-
stock-market-data-python-1/
"""

#%% ENV
import pandas as pd
# Package and modules for importing data;
# this code may change depending on pandas version
import pandas_datareader.data as web
import datetime
# Import matplotlib
import matplotlib.pyplot as plt
from CandleSticks import pandas_candlestick_ohlc


#%% SETTING
# This line is necessary for the plot to appear in a Jupyter notebook
%matplotlib inline
# Control the default size of figures in this Jupyter notebook
%pylab inline
pylab.rcParams['figure.figsize'] = (15, 9)   # Change the size of plots

# We will look at stock prices over the past year,
# starting at January 1, 2016
start = datetime.datetime(2016,1,1)
end = datetime.date.today()
 

#%% INPUT

# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source
# ("yahoo" for Yahoo! Finance), third is the start date,
# fourth is the end date
apple = web.DataReader("AAPL", "google", start, end)
 
type(apple)

# look at the table
apple.head()

# plot the close figures
apple["Close"].plot(grid = True) # Plot the adjusted closing price of AAPL

#%% CANDELSTICKS 
pandas_candlestick_ohlc(apple)


#%% 1+ STOCKS
microsoft = web.DataReader("MSFT", "google", start, end)
google = web.DataReader("GOOG", "google", start, end)
 
# Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
stocks = pd.DataFrame({"AAPL": apple["Close"],
                      "MSFT": microsoft["Close"],
                      "GOOG": google["Close"]})
 
stocks.head()

stocks.plot(grid = True)

stocks.plot(secondary_y = ["AAPL", "MSFT"], grid = True)

#%% COMPUTE RETURN
# df.apply(arg) will apply the function arg to each column in df, and return a DataFrame with the result
# Recall that lambda x is an anonymous function accepting parameter x; in this case, x will be a pandas Series object
stock_return = stocks.apply(lambda x: x / x[0])
stock_return.head()

stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
