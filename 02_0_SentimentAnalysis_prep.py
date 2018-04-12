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
import json
%matplotlib inline

print("---Enviroment---")
# %load_ext version_information
%reload_ext version_information
%version_information os, glob, panda, matplotlib, numpy, pprint, bs4, nltk
%version_information json

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
# there are jusy pos 4 and neg 0, so change pos to 1
df.loc[df.sentiment == 4, 'sentiment'] = 1
# heads
df[df.sentiment == 0].head(10)
df[df.sentiment == 1].head(10)

# drop the non relevant columns
df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

# sanity check, length of the string in text column in each entry.
df['pre_clean_len'] = [len(t) for t in df.text]

# %% PREP
print("---Prep---")
# Dictionary, first draft
data_dict = {
    'sentiment': {
        'type': df.sentiment.dtype,
        'description': 'sentiment class - 0:negative, 1:positive'
    },
    'text': {
        'type': df.text.dtype,
        'description': 'tweet text'
    },
    'pre_clean_len': {
        'type': df.pre_clean_len.dtype,
        'description': 'Length of the tweet before cleaning'
    },
    'dataset_shape': df.shape
}
pprint(data_dict)

# HTML Encoding
print("-----HTML Encoding")
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
example1 = re.sub('https?://[A-Za-z0-9./]+', '', df.text[0])
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
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't": "is not", "aren't": "are not",
                 "wasn't": "was not", "weren't": "were not",
                 "haven't": "have not", "hasn't": "has not",
                 "hadn't": "had not", "won't": "will not",
                 "wouldn't": "would not", "don't": "do not",
                 "doesn't": "does not", "didn't": "did not",
                 "can't": "can not", "couldn't": "could not",
                 "shouldn't": "should not", "mightn't": "might not",
                 "mustn't": "must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(
        lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above,
    # it has created unnecessary white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


# first 100 rows
print("Eg first 100 rows")
testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result


# record the time
% % time
print("Clean all the data")
# create an equally space  array in case we want to
# process by chunk
nums = np.linspace(start=0, stop=df.shape[0], num=10).astype(int)
print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
# for cycle
for i in range(nums[0], nums[-1]):
    if((i+1) % 10000 == 0):
        print("Tweets %d of %d has been processed" % (i+1, nums[-1]))
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

# %% PREP
print("---Output---")
print("Save the clean tweets and the sentiments")

# df
df_clean = pd.DataFrame(
    data=np.column_stack((clean_tweet_texts, df.sentiment.values)),
    columns=['text', 'target'])

df_clean.head()

df_clean.to_csv(
    "%s\\02_0_clean_tweet.csv" % data_directory,
    encoding='ISO-8859-1')
