"""
Fetch dataset

Documentation
-------------
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html?highlight=newsgroup#sklearn.datasets.fetch_20newsgroups
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import words

english_words = set(words.words())

_tokens_to_remove = stopwords.words('english')
_tokens_to_remove.append('e')
porter_stemmer = PorterStemmer()


def remove_stop_words(text, tokens_to_remove):
    """Remove common stop words from sentence"""
    new_text = ' '.join([x for x in text.split() if x not in tokens_to_remove])
    return new_text


def _stem(text, min_len_stemmed=2):
    """Remove stemming from sentence"""
    new_text = ' '.join([porter_stemmer.stem(x) for x in text.split()])
    new_text = ' '.join([x for x in new_text.split() if len(x) > min_len_stemmed])
    return new_text

def is_english(text):
    english_words_text = ' '.join([x for x in text.split() if x in english_words])
    return english_words_text

def process_text(input_text):
    # filtered.translate(str.maketrans('', '', string.punctuation))
    filtered = input_text.lower()
    filtered = re.sub('[^a-zA-Z]', ' ', filtered)
    filtered = re.sub(r'\[[0-9]*\]', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    # filtered = _stem(filtered)
    filtered = remove_stop_words(filtered,
                                 tokens_to_remove=_tokens_to_remove)
    filtered = is_english(filtered)

    return filtered

# PARAMS
remove_fields = ('headers', 'footers', 'quotes')
remove_fields = ()

# EXECUTE
train_bunch = fetch_20newsgroups(data_home=None,
                                 subset='train',
                                 remove=remove_fields)

X = train_bunch.data  # len(X) = 11314
# X = X[:1000]
x = X[0]  # len(x) = 475; otherwise 721 w/o remove fields
"""
sample_corpus = X[: sample_doc_size]
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(sample_corpus)
print(X_vect.toarray())
# x: 'I was wondering if anyone out there could enlighten me on ..."
vocab = vectorizer.get_feature_names()
print('len of vocab original: %d' % len(vocab))
"""

X_proc = [process_text(x) for x in X]  # list of sentences
vectorizer = CountVectorizer()
X_proc_vect = vectorizer.fit_transform(X_proc)
vocab = vectorizer.get_feature_names()  # list of words
print('len of vocab after processing: %d' % len(vocab))

S = X_proc_vect.toarray()
U = np.sum(S, axis=0)

dc = pd.DataFrame(U)  # len(U) =  num words in vocab
from matplotlib import pyplot as plt
dk = dc[dc <= dc.quantile(0.90)]
dk.hist(bins=20)
plt.xlabel('word appearances')
plt.ylabel('number of words')
plt.title('Distr of vocab <= 90% count quantile')



