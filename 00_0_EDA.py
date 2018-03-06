#%% DESCRIPTION
print("---Description---")
print("EDA Template")
print("Reference: \n  . https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python \n")

#%% ENV
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline

print("---Enviroment---")
#%load_ext version_information
%reload_ext version_information
%version_information pandas, matplotlib, seaborn, numpy, scipy, sklearn, numpy

#%% SETTING
print("---Setting---")
# set working directory
wd = os.path.abspath(os.path.dirname("__file__"))
print("The working directory is\n%s" % wd)
# set data directory
data_directory = os.path.abspath(os.path.join(wd, 'data', 'kaggle_housePrices'))
print("The data directory is\n%s" % data_directory)
# set files to load
csv_to_load = glob.glob("%s/*.csv" % data_directory)
print("The file to load are\n%s" % '\n'.join(csv_to_load))
# just extract the names to use as keys later
name_to_load = [f[len(data_directory)+1:-len(".csv")] for f in csv_to_load]

#%% INPUT
print('---Input---')
# read files in the folder and concat into one
df = pd.concat(
    [pd.read_csv(f, index_col='Id', header=0) for f in csv_to_load]
    , keys = name_to_load)

print('-> Info')
print(df.info())

"""
Have an excel with a table for data...
Variable - Variable name.
Type - variables' type.('numerical','categorical').
Segment - variables' segment.
Expectation - Our expectation about the variable influence the Y.
('High', 'Medium' and 'Low')
Conclusion - Our conclusions about the importance of the variable
after perform a little exploration.
('High', 'Medium' and 'Low')
Comments - Any general comments that occured to us.
"""

df_train['SalePrice'].describe()