import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize, pprint
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
from collections import Counter
import numpy as np


import glob


def data_cleaning(filepath):
    data=[]
    with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            data.append(line.strip())

    # drop the previous information (time||...)
    cleandata = []
    for x in data:
        cleandata.append(re.split('\|', x).pop(-1))

    df = pd.DataFrame({'col': cleandata})
    df = df['col'].str.replace('http\S+|www.\S+', '', case=False)
    df = df.str.lower()
    df = df.tolist()
    df = [''.join(c for c in s if c not in string.punctuation) for s in df]
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words)
    stop_words.extend(['video', 'audio', 'us', 'say', 'one', 'well', 'may', 'rt', 'says', 'new', 'get', '’'])
    word_token = []
    for i in df:
        word_token+=word_tokenize(i)

    filtered_sentence = [w for w in word_token if w not in stop_words]
    return filtered_sentence


def main():
    read_files = glob.glob("Health-Tweets/*.txt")
    list_of_lists=[]
    for f in read_files:
        list_of_lists.append(data_cleaning(f))

    model=Word2Vec(list_of_lists,min_count=1)

    X=model[model.wv.vocab]

    # pca
    reduced_tsvd=PCA(n_components=2)

    result=reduced_tsvd.fit_transform(X)
    #training model
    from sklearn.cluster import KMeans

    num_clusters = 4

    km = KMeans(n_clusters= num_clusters)

    km.fit(result)

    # clusters=km.labels_.tolist()

    y_kmeans=km.predict(result)

    #
    plt.scatter(result[:,0],result[:,1], c=y_kmeans)

    # cluster centers
    centers=km.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1], c='red')
    plt.savefig('clustering1.png')


    # similarity

if __name__=='__main__':
    main()
