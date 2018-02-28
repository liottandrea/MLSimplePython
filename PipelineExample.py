#%% DESCRIPTION
print("---Description---")
print("Example of DataframMapper and sklearn.pipeline")
print("Reference: https://www.giacomodebidda.com/sklearn-pandas/")
print("Reference: https://github.com/scikit-learn-contrib/sklearn-pandas")


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

print("---Enviroment---")
%load_ext version_information
%reload_ext version_information
%version_information os,glob, re, panda,sklearn, sklearn_pandas

#%% SETTING
print("---Setting---")

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
print('---Input---')
# read the two files and concat into one
df = pd.concat(
    [pd.read_csv(f, index_col='PassengerId', header=0) for f in csv_to_load]
    , keys = name_to_load)

print('-> Info')
print(df.info())

#%% FEATURE ENGINEERING
print('---Features Engineering---')
print('-> Describe')
print(df.describe())

print('-> Columns Overview')
for feature in set(df.columns.values).difference(set(['Name'])):
    print(feature)
    print(df[feature].value_counts(dropna = False))
    print('-' * 40)


print('->  DataFrameMapper: map colums to sklearn function')
mapper = DataFrameMapper([
    ('Pclass', None),
    ('Sex', LabelBinarizer()),
    (['Age'], [Imputer()]),
    ('SibSp', None, {'alias': 'Some variable'}),
    (['Ticket'], [LabelBinarizer()]),
    (['Fare'], Imputer()),
    ('Embarked', [CategoricalImputer(), MultiLabelBinarizer()]),
    ], default = False )

pipeline = Pipeline([
    ('feature_mapper', mapper),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=seed))
])

#%% PIPELINE
x_train = df.ix["train"]
x_train[x_train.columns.drop(['Survived'])]

y_train = df_train['Survived']

# one way of computing cross-validated accuracy estimates
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
scores = cross_val_score(pipeline, x_train, y_train, cv=skf)
print('Accuracy estimates: {}'.format(scores))

    # another way of computing cross-validated accuracy estimates
for i_split, (ii_train, ii_test) in enumerate(skf.split(X=x_train, y=y_train)):
    # x_train (independent variables, aka features) is a pandas DataFrame.
    # xx_train and xx_test are pandas dataframes
    # xx_train = x_train.iloc[ii_train, :]
    xx_test = x_train.iloc[ii_test, :]
    # y_train (target variable) is a pandas Series.
    # yy_train and yy_test are numpy arrays
    yy_train = y_train.values[ii_train]
    yy_test = y_train.values[ii_test]

    model = pipeline.fit(X=xx_train, y=yy_train)
    predictions = model.predict(xx_test)
    score = accuracy_score(y_true=yy_test, y_pred=predictions)
     print('Accuracy of split num {}: {}'.format(i_split, score))

# final model (retrain it on the entire train set)
model = pipeline.fit(X=x_train, y=y_train)

# In this problem df_test doesn't contain the target variable 'Survived'
x_test = df_test
predictions = model.predict(x_test)
print('Predictions (0 = dead, 1 = survived)')
print(predictions)

