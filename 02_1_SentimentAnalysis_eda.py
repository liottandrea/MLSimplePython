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
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from pylab import *
%matplotlib inline

print("---Enviroment---")
# %load_ext version_information
%reload_ext version_information
%version_information os, glob, panda, matplotlib, numpy, pprint, bs4, nltk
%version_information wordcloud, sklearn

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
my_df = pd.read_csv(
    csv_to_load[2], index_col=0,
    encoding='"ISO-8859-1"',
    engine='c')

# checks
print("Checks")
my_df.head()
my_df.info()
print("Null values?")
my_df[my_df.isnull().any(axis=1)].head()
np.sum(my_df.isnull().any(axis=1))
# By looking these entries in the original data,
# it seems like only text information they had was either twitter ID or
# url address. Anyway, these are the info I decided to discard for the
# sentiment analysis, so I will drop these null rows, and update
# the data frame.
print("Delete Null values")
my_df.dropna(inplace=True)
my_df.reset_index(drop=True, inplace=True)
my_df.info()

print("---WordCloud---")
print("Negative Sentiment")
neg_tweets = my_df[my_df.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,
                      =200).generate(neg_string)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

print("love word ?")
for t in neg_tweets.text[:200]:
    if 'love' in t:
        print(t)

print("Positive Sentiment")
pos_tweets = my_df[my_df.target == 1]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)

pos_string = pd.Series(pos_string).str.cat(sep=' ')
wordcloud = WordCloud(
    width=1600,
    height=800
    =200,
    colormap='magma').generate(pos_string)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


print("---Term Frequency---")
cvec = CountVectorizer()
cvec.fit(my_df.text)
print("words count")
len(cvec.get_feature_names())


neg_doc_matrix = cvec.transform(my_df[my_df.target == 0].text)
pos_doc_matrix = cvec.transform(my_df[my_df.target == 1].text)
neg_tf = np.sum(neg_doc_matrix, axis=0)
pos_tf = np.sum(pos_doc_matrix, axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame(
    [neg, pos], columns=cvec.get_feature_names()).transpose()

# add col total
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

print("---Zipf Law---")
y_pos = np.arange(500)
plt.figure(figsize=(10, 8))
s = 1
expected_zipf = [
    term_freq_df.sort_values(
        by='total',
        ascending=False)['total'][0] / (i+1)**s for i in y_pos]
plt.bar(
    y_pos,
    term_freq_df.sort_values(
        by='total',
        ascending=False)['total'][:500],
    align='center', alpha=0.5)
plt.plot(y_pos, expected_zipf, color='r',
         linestyle='--', linewidth=2, alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in tweets')


counts = term_freq_df.total
tokens = term_freq_df.index
ranks = arange(1, len(counts)+1)
indices = argsort(-counts)
frequencies = counts[indices]
plt.figure(figsize=(8, 6))
plt.ylim(1, 10**6)
plt.xlim(1, 10**6)
loglog(ranks, frequencies, marker=".")
plt.plot([1, frequencies[0]], [frequencies[0], 1], color='r')
title("Zipf plot for tweets tokens")
xlabel("Frequency rank of token")
ylabel("Absolute frequency of token")
grid(True)
for n in list(logspace(-0.5, log10(len(counts)-2), 25).astype(int)):
    dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]],
                 verticalalignment="bottom",
                 horizontalalignment="left")


# https: // towardsdatascience.com/
# another-twitter-sentiment-analysis-with-python-
# part-3-zipfs-law-data-visualisation
