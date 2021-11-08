"""
Preprocessing of train and test newsgroups data.

DataSet Documentation
-------------
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html?highlight=newsgroup#sklearn.datasets.fetch_20newsgroups


@author: ivbarrie

"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups

from nltk.corpus import words as nltk_english_words



def remove_stop_words(text, tokens_to_remove):
    """Remove common stop words from sentence"""
    new_text = ' '.join([x for x in text.split() if x not in tokens_to_remove])
    return new_text


def _stem(text, min_len_stemmed=2):
    """Remove stemming from sentence"""
    new_text = ' '.join([porter_stemmer.stem(x) for x in text.split()])
    new_text = ' '.join([x for x in new_text.split() if len(x) > min_len_stemmed])
    return new_text


def is_english(text, english_words):
    english_words_text = ' '.join([x for x in text.split() if x in english_words])
    return english_words_text


def process_text(input_text, tokens_to_remove):
    # filtered.translate(str.maketrans('', '', string.punctuation))
    filtered = input_text.lower()
    filtered = re.sub('[^a-zA-Z]', ' ', filtered)
    filtered = re.sub(r'\[[0-9]*\]', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    # filtered = _stem(filtered)
    filtered = remove_stop_words(filtered,
                                 tokens_to_remove=tokens_to_remove)
    filtered = is_english(filtered)

    return filtered


class PreProcessor:
    """Preprocessing of text"""

    def __init__(self, train_dest, test_dest,
                 remove_fields=('headers', 'footers', 'quotes')):

        self.train_dest = train_dest
        self.test_dest = test_dest
        self.remove_fields = remove_fields
        self.port_stemmer = PorterStemmer()
        self.english_words = set(nltk_english_words.words())
        self._tokens_to_remove = stopwords.words('english')
        self._tokens_to_remove.append('e')
        self.train_data = [""]
        self.test_data = [""]
        self.count_vectorizer = None  # def in train data processing
        self.vocab = None  # def in training data processing

    def load_data(self, subset=''):
        """Load 20 Newsgroup train data"""
        if subset not in ['train', 'test']:
            raise Exception("subset of 20NewsGroup must be 'train' or 'test'")
        data_bunch = fetch_20newsgroups(data_home=None,
                                        subset=subset,
                                        remove=self.remove_fields)

        return data_bunch

    def set_train_data(self):
        """Set raw train data"""
        train_bunch = self.load_data(subset='train')
        self.train_data = train_bunch.data  # List[str]

    def set_test_data(self):
        """Set raw test data"""
        test_bunch = self.load_data(subset='test')
        self.test_data = test_bunch.data  # List[str]

    def preprocess_data_to_array(self, subset='train') -> np.ndarray:
        """Preprocess text to 2-d array of word counts"""
        if subset == 'train':
            self.count_vectorizer = CountVectorizer()
            data = self.train_data
            x_proc = [process_text(x) for x in data]  # list of sentences
            x_proc_vect = self.count_vectorizer.fit_transform(x_proc)  # scipy.sparse.csr.csr_matrix
            self.vocab = self.count_vectorizer.get_feature_names()  # List[str]
        else:
            data = self.test_data
            # data = [" ".join(i for i in w.split() if i in self.vocab) for w in data]
            x_proc = [process_text(x) for x in data]
            x_proc_vect = self.count_vectorizer.transform(x_proc)
        data_vect_array = x_proc_vect.toarray()
        return data_vect_array




