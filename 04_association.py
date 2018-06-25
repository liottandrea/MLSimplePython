#%% DESCRIPTION
# Clustering Template
# standard template to run classification in Python

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
# Training Apriori on the dataset
from myApriori import apriori
# extra functions
import dataInOut as myio

print("---Enviroment---")

%load_ext version_information
%reload_ext version_information
%version_information os,glob, numpy, panda,sklearn, statsmodels

#%% SETTING
print("---Setting---")
setting = {
        # seed for random
        "seed_random": 0,
        # test size
        "test_size": 0.25,
        # input to use
        'which_input': 'Apriori'}
print(setting)

# we have dictionary with the setting for each input
which_input = {
# Market_Basket_Optimisation - Apriori
        'Apriori' : {
                'csv_file':'data/Market_Basket_Optimisation.csv'}}
# select the one that you want to run
print("---Setting Input---")
setting_input = which_input[setting['which_input']]
print(setting_input)

#%% INPUT

# from csv to two datasets
dataset = pd.read_csv(setting_input['csv_file'], header = None)



transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append(
            [str(dataset.values[i,j])
            for j in range(0, dataset.shape[1])])


# min_support    
# weekly purchase, I want to check product
# buy more than 3 times a day    
3 * 7 / dataset.shape[0]
  
  
# Training Apriori on the dataset
rules = apriori(transactions,
                min_support = 0.003,
                min_confidence = 0.2,
                min_lift = 3,
                min_length = 2)

# Visualising the results
results = list(rules)
