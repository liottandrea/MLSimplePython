#%% DESCRIPTION
# Regression Template
# standard template to run regression in Python

#%% ENV

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data partition
from sklearn.model_selection import train_test_split
# Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# for linear regression
from sklearn.linear_model import LinearRegression
# linear regression
import statsmodels.formula.api as sm
# poly variables
from sklearn.preprocessing import PolynomialFeatures
# scaling
from sklearn.preprocessing import StandardScaler
# pandas and sklearn working together
# https://github.com/scikit-learn-contrib/sklearn-pandas
# from sklearn_pandas import DataFrameMapper
# SVR
from sklearn.svm import SVR
# Decision tree
from sklearn.tree import DecisionTreeRegressor
# Random forecast
from sklearn.ensemble import RandomForestRegressor
# extra functions
import dataInOut as myio

print("---Enviroment---")
'''
%load_ext version_information
%reload_ext version_information
%version_information os,glob, numpy, panda,sklearn, sklearn_pandas, statsmodels
'''
#%% SETTING
print("---Setting---")
setting = {
        # seed for random
        "seed_random": 0,
        # test size
        "test_size": 0.2,
        # input to use
        'which_input': 'SvmReg'}
print(setting)

# we have dictionary with the setting for each input
# 50_Startups - multiple linear
which_input = {
        'MultiLinearReg' : {
                'csv_file':'data/50_Startups.csv',
                'y_col':'Profit',
                'X_cols':['R&D Spend', 'Administration',
                        'Marketing Spend', 'State'],
                'dummy_cols':'State'},
# Salary_Data - simple linear
        'SimpleLinearReg': {
                'csv_file':'data/Salary_Data.csv',
                'y_col':'Salary',
                'X_cols':'YearsExperience'},
# 'Position_Salaries.csv' for poly
        'PolyReg': {
                'csv_file':'data/Position_Salaries.csv',
                'y_col':'Salary',
                'X_cols':'Level',
                'poly_degree': 4},
# 'Position_Salaries.csv' for svm
        'SvmReg': {
                'csv_file':'data/Position_Salaries.csv',
                'y_col':'Salary',
                'X_cols':'Level',
                'feature_scaling':True}
}

# select the one that you want to run
print("---Setting Input---")
setting_input = which_input[setting['which_input']]
print(setting_input)

#%% INPUT

# from csv to two datasets
X, y = myio.csv_df2Xy(
        setting_input['csv_file'],
        setting_input['X_cols'],
        setting_input['y_col'])

#%% PREPROCESSING

# Feature Scaling
if 'feature_scaling' in setting_input:
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

# DUMMY VARIABLES
if 'dummy_cols' in setting_input:
    X = pd.get_dummies(
            X,
            columns = [setting_input['dummy_cols']],
            drop_first = True) # Avoiding the Dummy Variable Trap

# POLYNOMIAL
if 'poly_degree' in setting_input:
    # set polynomial
    poly_transform = PolynomialFeatures(
            degree = setting_input['poly_degree'],
            include_bias = False)
    # fit it
    poly_transform.fit_transform(X)
    # put in a df
    X = pd.DataFrame(
            data = poly_transform.fit_transform(X),
            columns = poly_transform.get_feature_names(X.columns),
            index = X.index)
    
# INTERCEPT
X = pd.DataFrame(
        data = PolynomialFeatures(degree = 1).fit_transform(X),
        columns = ['Intercept'] + list(X),
        index = X.index)


#%% PARTITION

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = setting['test_size'],
        random_state = setting['seed_random'])


#%% REGRESSION simple, multi, poly

# FIT
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train, y_train)


#%% SV REGRESSION
# FIT
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#%% DECISION TREE REGRESSION
# FIT
regressor = DecisionTreeRegressor(random_state = setting['seed_random'])
regressor.fit(X, y)

#%% RANDOM FORECAST REGRESSION
# FIT
regressor = RandomForestRegressor(n_estimators = 10, random_state = setting['seed_random'])
regressor.fit(X, y)


# %% PREDICT
# fitted values on train set
y_train_pred = pd.DataFrame(data = regressor.predict(X_train),
                      columns = list(y_train),
                      index = X_train.index)

# fitted values on test set
y_test_pred = pd.DataFrame(data = regressor.predict(X_test),
                      columns = list(y_train),
                      index = X_test.index)


#%% VISUALISE

# Simple linear regression
## Visualising the Training set results
plt.scatter(X_train[setting_input['X_cols']].values, y_train.values, color = 'red')
plt.plot(X_train[setting_input['X_cols']].values, y_train_pred.values, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Visualising the Test set results
plt.scatter(X_test[setting_input['X_cols']].values, y_test.values, color = 'red')
plt.plot(X_train[setting_input['X_cols']].values, y_train_pred.values, color = 'blue')
plt.scatter(X_test[setting_input['X_cols']].values, y_test_pred.values, color = 'orange')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Polynomial Regression 
## Visualising the Polynomial Regression results
## (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid,
         regressor.predict(poly_transform.fit_transform(X_grid)),
         color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




#%% MANUAL SEARCH BEST REGRESSION
# Building the optimal model using Backward Elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
