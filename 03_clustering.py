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
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
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
        'which_input': 'KMeans'}
print(setting)

# we have dictionary with the setting for each input
which_input = {
# Mall_Customers - Kemans 
        'KMeans' : {
                'csv_file':'data/Mall_Customers.csv',
                'X_cols':['Annual Income (k$)', 'Spending Score (1-100)'],
                'y_col':['Annual Income (k$)']}
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
     
    
   
#%% CLUSTERING Kmeans
# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    random_state = setting['seed_random'])
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = setting['seed_random'])
y_kmeans = kmeans.fit_predict(X)



#%% VISUALISE Kmeans

X_2 = X.loc[:,setting_input['X_cols']].values

# Visualising the clusters
plt.scatter(X_2[y_kmeans == 0, 0], X_2[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_2[y_kmeans == 1, 0], X_2[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_2[y_kmeans == 2, 0], X_2[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_2[y_kmeans == 3, 0], X_2[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_2[y_kmeans == 4, 0], X_2[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

   
#%% CLUSTERING HC
# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

X_2 = X.loc[:,setting_input['X_cols']].values

# Visualising the clusters
plt.close()
plt.scatter(X_2[y_hc == 0, 0], X_2[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X_2[y_hc == 1, 0], X_2[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X_2[y_hc == 2, 0], X_2[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X_2[y_hc == 3, 0], X_2[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X_2[y_hc == 4, 0], X_2[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


