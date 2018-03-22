# %% ENV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pprint import pprint
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
%matplotlib inline

print("---Enviroment---")
# %load_ext version_information
%reload_ext version_information
%version_information os, glob, panda, matplotlib, numpy, pprint, bs4, nltk

# %% DESCRIPTION
print("---Description---")
print("Twitter Sentiment Analysis")
print("Goal: classify tweets by sentitment")
print("Reference:")
print("http://help.sentiment140.com/for-students/")


# %% SETTING
print("---Setting---")
# set working directory
wd = os.path.abspath(os.path.dirname("__file__"))
print("The working directory is\n%s" % wd)
# set data directory
data_directory = os.path.abspath(os.path.join(wd, 'data', 'Sentiment140'))
print("The data directory is\n%s" % data_directory)
# set files to load
csv_to_load = glob.glob("%s/*.csv" % data_directory)
print("The file to load are\n%s" % '\n'.join(csv_to_load))
# just extract the names to use as keys later
name_to_load = [f[len(data_directory)+1:-len(".csv")] for f in csv_to_load]


csv_to_load[1]
# %% INPUT
print("---Input---")
cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
df = pd.read_csv(
    csv_to_load[1],
    header=None,
    names=cols,
    encoding='"ISO-8859-1"',
    engine='c')

# First check
df.head()
df.sentiment.value_counts()
df[df.sentiment == 0].head(10)
df[df.sentiment == 4].head(10)

# drop the non relevant columns
df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

# %% PREP
print("---Prep---")
# Dictionary, first draft
data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}
pprint(data_dict)

# HTML Encoding
print("-----HTML Encoding")
# sanity check, length of the string in text column in each entry.
df['pre_clean_len'] = [len(t) for t in df.text]
# plot it
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()
# tweets nchar has a limit on 140, it appear that we have
# an issue on html enconding of special character
df[df.pre_clean_len > 140].head(10)
# eg
print(df.text[279])
# convert it
example1 = BeautifulSoup(df.text[279], 'lxml')
print(example1.get_text())

# @mention
print("-----@mention")
# @mention carries a certain information
# (which another user that the tweet mentioned),
# this information doesn’t add value to build sentiment analysis model.
# eg
print(df.text[343])
# convert it
example1 = re.sub(r'@[A-Za-z0-9]+', '', df.text[343])
print(example1)


# @mention
print("-----URL links")
# The third part of the cleaning is dealing with URL links
# eg
print(df.text[0])
# convert it
example1 = re.sub('https?://[A-Za-z0-9./]+','',df.text[0])
print(example1)


# UTF-8 BOM
print("-----UTF-8 BOM")
# strange character
# eg
print(df.text[226])
# convert it
example1 = df.text[226].encode('latin-1').decode('utf-8-sig')
print(example1)


# hashtag / numbers
print("-----hashtag / numbers")
# eg
print(df.text[175])
# convert it
example1 = re.sub("[^a-zA-Z]", " ", df.text[175])
print(example1)


# Defining data cleaning function
print("-----Data Cleaning")
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.encode('latin-1').decode('utf-8-sig')
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above,
    # it has created unnecesary white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()


testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result
