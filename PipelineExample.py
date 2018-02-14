#%% DESCRIPTION
"""
Run through some example of using sklearn-pandas and sklearn pipeline
https://www.giacomodebidda.com/sklearn-pandas/
"""

#%% ENV
import os
import glob
import re
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, Imputer, LabelEncoder, \
    FunctionTransformer, Binarizer, StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn_pandas import DataFrameMapper, CategoricalImputer

#%% SETTING

# set working directory
wd = os.path.abspath(os.path.dirname("__file__"))
print("The working directory is\n%s" % wd)

# set data directory
data_directory = os.path.abspath(os.path.join(wd, 'data', 'kaggle_titanic'))
print("The data directory is\n%s" % data_directory)

# set files to load
csv_to_load = glob.glob("%s/*.csv" % data_directory)
print("The file to load are\n%s" % '\n'.join(csv_to_load))
# just extract the names to use as keys later
name_to_load = [f[len(data_directory)+1:-len(".csv")] for f in csv_to_load]

#%% INPUT
df = pd.concat(
    [pd.read_csv(f, index_col='PassengerId', header=0) for f in csv_to_load]
    , keys = name_to_load)


print('--- Info ---')
print(df.info())
print('--- Describe ---')
print(df.describe())

print('--- Features ---')
for feature in set(df_train.columns.values).difference(set(['Name'])):
    print(feature)
    print(df[feature].value_counts(dropna=False))
    print('-' * 40)