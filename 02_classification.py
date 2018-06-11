#%% DESCRIPTION
# Classification Template
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
# Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
# confusion matrix
from sklearn.metrics import confusion_matrix
# colors for plot
from matplotlib.colors import ListedColormap
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
        'which_input': 'Logistic'}
print(setting)

# we have dictionary with the setting for each input
which_input = {
# Social_Network_Ads - logistic 
        'Logistic' : {
                'csv_file':'data/Social_Network_Ads.csv',
                'y_col':'Purchased',
                'X_cols':['Age','EstimatedSalary'], #Gender
              #  'dummy_cols':'Gender',
                'feature_scaling':['Age','EstimatedSalary']}
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

# DUMMY VARIABLES
if 'dummy_cols' in setting_input:
    X = pd.get_dummies(
            X,
            columns = [setting_input['dummy_cols']],
            drop_first = True) # Avoiding the Dummy Variable Trap



#%% PARTITION

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = setting['test_size'],
        random_state = setting['seed_random'])

#%% SCALING
# Feature Scaling
if 'feature_scaling' in setting_input:
    sc_X = StandardScaler()
    sc_X_fit = sc_X.fit(X_train[setting_input['feature_scaling']].values)
    # train
    sc_X_transform = sc_X_fit.transform(X_train[setting_input['feature_scaling']].values)
    X_train.loc[:,setting_input['feature_scaling']] = sc_X_transform 
    # test
    sc_X_transform = sc_X_fit.transform(X_test[setting_input['feature_scaling']].values)
    X_test.loc[:,setting_input['feature_scaling']] = sc_X_transform 
     
    
   
#%% REGRESSION

# FIT
# Fitting Logistic Regression to the Training set
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier = SVC(kernel = 'linear', random_state = setting['seed_random'])
classifier = SVC(kernel = 'rbf', random_state = setting['seed_random'])
classifier = SVC(kernel = 'poly', random_state = setting['seed_random'])
classifier = GaussianNB()
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = setting['seed_random'])
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = setting['seed_random'])
classifier = LogisticRegression(random_state = setting['seed_random'])
classifier.fit(X_train, y_train)

# %% PREDICT
# fitted values on train set
y_train_pred = pd.DataFrame(data = classifier.predict(X_train),
                      columns = list(y_train),
                      index = X_train.index)

# fitted values on test set
y_test_pred = pd.DataFrame(data = classifier.predict(X_test),
                      columns = list(y_train),
                      index = X_test.index)



#%% VISUALISE

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)


# Visualising the Training set results
X_set, y_set = X_train.values, y_train.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('#FFA07A', '#90EE90')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
            X_train.loc[y_train[y_train[setting_input['y_col']] == i].index]['Age'].values,
            X_train.loc[y_train[y_train[setting_input['y_col']] == j].index]['EstimatedSalary'].values,
            c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
