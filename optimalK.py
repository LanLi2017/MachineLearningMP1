import re

from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import string

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

# find the optimal k
def Kcluster(result):
    wcc =[]
    for num_clusters in range(1,11):

        km=KMeans(num_clusters)
        km.fit(result)
        wcc.append(km.inertia_)
    import matplotlib.pyplot as plt
    plt.scatter(range(1,11),wcc)
    plt.title("Elbow Method")
    plt.savefig('Elbow.png')


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
    Kcluster(result)



    # similarity

if __name__=='__main__':
    main()
