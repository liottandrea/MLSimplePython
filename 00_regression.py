#%% DESCRIPTION

# Regression Template

#%% ENV

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# data partition
from sklearn.model_selection import train_test_split
# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# for linear regression
from sklearn.linear_model import LinearRegression
# linear regression
import statsmodels.formula.api as sm
# poly variables
from sklearn.preprocessing import PolynomialFeatures

# functions
def csv2dfs(csv_file,x_cols,y_col):
    # read csv
    dataset = pd.read_csv(csv_file)
    # divide x and y
    X = dataset[x_cols]
    y = dataset[y_col]
    # force to datafram in case only one column
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    # return
    return X,y


#%% SETTING

# seed for random
seed_random = 0
# test size
test_size = 0.2

# IN
which_input = 'startups'
which_input = 'salary_linear'


# 50_Startups - multiple linear
if which_input == 'startups': 
    input_setting = {
            'csv_file':'data/50_Startups.csv',
            'y_col':'Profit',
            'X_cols':['R&D Spend', 'Administration',
                      'Marketing Spend', 'State'],
            'dummy_cols':'State'}

# Salary_Data - simple linear
if which_input == 'salary_linear': 
    input_setting = {
            'csv_file':'data/Salary_Data.csv',
            'y_col':'Salary',
            'X_cols':['YearsExperience'],
            'dummy_cols':np.nan}

#%% INPUT

# from csv to two datasets
X, y = csv2dfs(input_setting['csv_file'],
                    input_setting['X_cols'],
                    input_setting['y_col'])


#%% PREPROCESSING

# DUMMY VARIABLES
if isinstance(input_setting['dummy_cols'],list):
    X = pd.get_dummies(
            X,
            columns = [input_setting['dummy_cols']],
            drop_first = True) # Avoiding the Dummy Variable Trap

# INTERCEPT
X = pd.DataFrame(
        data = PolynomialFeatures(degree = 1).fit_transform(X),
        columns = ['Intercept'] + list(X),
        index = X.index)


#%% PARTITION

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = test_size,
        random_state = seed_random)


#%% LINEAR REGRESSION

# FIT
# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train, y_train)

# PREDICT
# fitted values on train set
y_train_pred = pd.DataFrame(data = regressor.predict(X_train),
                      columns = list(y_train),
                      index = X_train.index)

# fitted values on test set
y_test_pred = pd.DataFrame(data = regressor.predict(X_test),
                      columns = list(y_train),
                      index = X_test.index)


#%% POLYNOMIAL REGRESSION




#%% VISUALISE

# Simple linear regression
## Visualising the Training set results
plt.scatter(X_train[input_setting['X_cols']].values, y_train.values, color = 'red')
plt.plot(X_train[input_setting['X_cols']].values, y_train_pred.values, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Visualising the Test set results
plt.scatter(X_test[input_setting['X_cols']].values, y_test.values, color = 'red')
plt.plot(X_train[input_setting['X_cols']].values, y_train_pred.values, color = 'blue')
plt.scatter(X_test[input_setting['X_cols']].values, y_test_pred.values, color = 'orange')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()





