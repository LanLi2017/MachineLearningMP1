import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import numpy as np

# list the txt
data = []
import glob

read_files = glob.glob("Health-Tweets/*.txt")

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())


with open('result.txt',encoding='utf8',errors='ignore') as f:
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
stop_words.extend(['video', 'audio','us','say','one','well','may','rt','says','new','get','’'])
word_token = []
for i in df:
    word_token += word_tokenize(i)
# remove the stopwords for
filtered_sentence = [w for w in word_token if w not in stop_words]


counts = Counter(filtered_sentence).most_common(10)
print(counts)

df_count = pd.DataFrame(counts, columns=['Word','Count'])
print(df_count)

df_count.plot.bar(x='Word',y = 'Count', figsize=(10,12))

plt.savefig('Histogram.png')

# str1 =  (', '.join(filtered_sentence))
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110,background_color='white').generate(str1)
#
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.savefig('wordcloud.png')


#

