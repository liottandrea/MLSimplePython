# %% ENV
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

# %% DESCRIPTION
print("---Description---")
print("EDA Template")
print("Reference:")
print(". https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python")

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
data_directory = os.path.abspath(os.path.join(wd, 'data', 'kaggle_housePrices'))
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
    [pd.read_csv(f, index_col='Id', header=0) for f in csv_to_load],
    keys=name_to_load)

# divide train and test
df_train = df.loc["train"]
df_test = df.loc["test"]

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
# %% EDA y
y_col = 'SalePrice'

print('-> y: %s' % y_col)
df_train[col_y].describe()
print('--> Histogram')
sns.distplot(df_train[y_col])
"""
Deviate from the normal distribution
Have appreciable positive skewness
Show peakedness
"""
print('--> Other Metrics')
print("Skewness: %f" % df_train[y_col].skew())
print("Kurtosis: %f" % df_train[y_col].kurt())

# %% y VS X
print('-> y and X (continuous)')
# SalePrice vs GrLivArea
x_col = 'GrLivArea'
print('--> Scatterplot %s & %s' % (y_col, x_col))
df_train.plot.scatter(x=x_col, y=y_col)
# SalePrice vs TotalBsmtSF
x_col = 'TotalBsmtSF'
print('--> Scatterplot %s & %s' % (y_col, x_col))
df_train.plot.scatter(x=x_col, y=y_col)

print('-> y and X (categorical)')
x_col = 'OverallQual'
print('--> Boxplot %s & %s' % (y_col, x_col))
# increase the size of the plot
plt.subplots(figsize=(8, 6))
sns.boxplot(x=x_col, y=y_col, data=df_train.loc[:, [x_col, y_col]])

x_col = 'YearBuilt'
print('--> Boxplot %s & %s' % (y_col,x_col))
plt.subplots(figsize=(16, 8))
# rotare the x ticks
plt.xticks(rotation=90)
fig = sns.boxplot(x = x_col, y = y_col, data = df_train.loc[:,[x_col, y_col]])


#%% The Dataset
print('-> The Dataset')
print('--> Correlation Matrix')
correlation_matrix = df_train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(correlation_matrix, vmax = .8, square = True)

print('--> y Correlation Matrix')
# number of variables for heatmap
k = 10
# the k variables with highest correlation
name_col = correlation_matrix.nlargest(k, y_col)[y_col].index
correlation_matrix_subset = np.corrcoef(df_train[name_col].values.T)
sns.set(font_scale=1.25)
sns.heatmap(correlation_matrix_subset,
                 cbar=True, annot=True,
                 square=True, fmt='.2f',
                 annot_kws={'size': 10},
                 yticklabels = name_col.values, xticklabels = name_col.values)


print('--> Scatterplot Matrix')
# clean the setting
sns.set()
name_col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[name_col], size = 2.5)


#%% Missing Data
print('->  Missing Data')
print('--> Missing by columns top 20')
# count missing data
missing_data_count = df_train.isnull().sum().sort_values(ascending=False)
# compute percent
missing_data_percent = missing_data_count/len(df_train.index)
# concat and print
missing_data = pd.concat(
    [missing_data_count, missing_data_percent],
    axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))

# Delete var with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
# Delete a row with missing data in Electrical
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#%% Outliers
print('->  Outliers')
print('--> Scale y ~ (mena = 0, sd = 1)')
df_train = df_train.join(pd.DataFrame(
    StandardScaler().fit_transform(df_train[[y_col]]),
    index = df_train.index,
    columns = ['%s_scaled'% y_col]))
print('---> outer range (low) of the distribution')

print(df_train.sort_values(by='%s_scaled'% y_col).head(10))

print('---> outer range (high) of the distribution')

print(df_train.sort_values(by='%s_scaled'% y_col).tail(10))

#%% Distributions
print('-> Distributions')
print('--> is y normal?')
sns.distplot(df_train[y_col], fit = norm)
fig = plt.figure()
stats.probplot(df_train[y_col], plot = plt)

print('--> apply log transform')
df_train['%s_log'% y_col] = np.log(df_train[y_col])
print('--> is log(y) normal?')
sns.distplot(df_train['%s_log'% y_col], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train['%s_log'% y_col], plot=plt)

x_col = 'TotalBsmtSF'
print('--> is %s normal?' % x_col)
sns.distplot(df_train[x_col], fit = norm)
fig = plt.figure()
stats.probplot(df_train[x_col], plot = plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

print('--> apply log transform only on strict positive values')
df_train.loc[df_train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

print('--> is %s normal? with the log transform (positive values only)' % x_col)
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)


#%% Homoscedasticity

#%% Dummies
df_train = pd.get_dummies(df_train)
