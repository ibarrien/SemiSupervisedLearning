"""
Preprocessing of train and test newsgroups data.

DataSet Documentation
-------------
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html?highlight=newsgroup#sklearn.datasets.fetch_20newsgroups

Example of working with 20 Newsgroup data from:
https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html

# TRAIN
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)

# TEST
newsgroups_test = fetch_20newsgroups(subset='test',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')  # 0.77


@author: ivbarrie

"""

import re
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.datasets import fetch_20newsgroups


def remove_stop_words(text: str, tokens_to_remove: List[str]) -> str:
    """Remove common stop words from sentence"""
    new_text = ' '.join([x for x in text.split() if x not in tokens_to_remove])
    return new_text


def _stem(text: str, min_len_stemmed: int = 2) -> str:
    """Remove stemming from sentence"""
    new_text = ' '.join([porter_stemmer.stem(x) for x in text.split()])
    new_text = ' '.join([x for x in new_text.split() if len(x) > min_len_stemmed])
    return new_text


def is_english(text: str, english_vocab: List[str]) -> str:
    """Remove words not in english vocab."""
    english_words_text = ' '.join([x for x in text.split() if x in english_vocab])
    return english_words_text


def process_text(text: str, tokens_to_remove: List[str], english_vocab: List[str]) -> str:
    """Basic text processing."""
    # filtered.translate(str.maketrans('', '', string.punctuation))
    filtered = text.lower()
    filtered = re.sub('[^a-zA-Z]', ' ', filtered)
    filtered = re.sub(r'\[[0-9]*\]', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)
    # filtered = _stem(filtered)
    filtered = remove_stop_words(filtered, tokens_to_remove=tokens_to_remove)
    filtered = is_english(filtered, english_vocab=english_vocab)
    return filtered


class TextPreProcessor:
    """Preprocessing of text documents."""

    def __init__(self, tokens_to_remove, english_vocab,
                 doc_axis: int = 1,
                 label_names_key: str = 'target_names',
                 label_vals_key: str = 'target',
                 remove_zero_vocab_docs: bool = True,
                 remove_fields=('headers', 'footers', 'quotes')):
        self.tokens_to_remove = tokens_to_remove
        self.english_vocab = english_vocab
        self.doc_axis = doc_axis
        self.remove_fields = remove_fields
        self.port_stemmer = PorterStemmer()
        self.label_names_key = label_names_key
        self.label_vals_key = label_vals_key
        self.remove_zero_vocab_docs = remove_zero_vocab_docs
        self.train_data = [""]
        self.train_label_vals = [0]
        self.train_count_data = [0]
        self.test_data = [""]
        self.test_label_vals = [0]
        self.test_count_data = [0]
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

    def set_train_raw_data(self):
        """Set raw train data"""
        self.train_bunch = self.load_data(subset='train')
        self.train_data = self.train_bunch.data  # List[str]
        self.train_label_vals = self.train_bunch[self.label_vals_key]  # List[str], len = len(self.train_data)
        self.train_label_names = self.train_bunch[self.label_names_key]  # List[str], len = 20

    def set_test_raw_data(self):
        """Set raw test data"""
        test_bunch = self.load_data(subset='test')
        self.test_data = test_bunch.data  # List[str]

    def process_text(self, text_list: List[str]) -> List[str]:
        """Apply basic text pre-processing to loaded data."""
        x_proc = [process_text(x, tokens_to_remove=self.tokens_to_remove, english_vocab=self.english_vocab)
                  for x in text_list]  # list of sentences
        return x_proc

    def preprocess_data_to_array(self, subset: str = 'train') -> np.ndarray:
        """Preprocess text to 2-d array of word counts via count vectorization.
        Rows: document index; Columns: word count index
        """
        if subset == 'train':
            self.count_vectorizer = CountVectorizer()
            if not self.train_data or len(self.train_data) < 2:
                raise Exception("(raw) train data not set; run set_train_raw_data()")
            data = self.train_data
            x_proc = self.process_text(text_list=data)
            x_proc_vect = self.count_vectorizer.fit_transform(x_proc)  # scipy.sparse.csr.csr_matrix
            self.vocab = self.count_vectorizer.get_feature_names()  # List[str]
        else:
            data = self.test_data
            if not self.test_data or len(self.test_data) < 2:
                raise Exception("(raw) test data not set; run\n set_test_raw_data()")
            x_proc = self.process_text(text_list=data)
            x_proc_vect = self.count_vectorizer.transform(x_proc)
        data_vect_array = x_proc_vect.toarray()
        return data_vect_array

    @staticmethod
    def remove_zero_count_docs(doc_count_data: np.ndarray, doc_axis: int = 1) -> Tuple[np.ndarray]:
        """Remove zero count doc vectors wrt *preprocessed vocab* and keep their labels."""
        count_sums = np.sum(doc_count_data, axis=doc_axis)
        mask = count_sums > 0  # use original indices for label val retrieval
        nonzero_doc_data = doc_count_data[count_sums > 0]
        return nonzero_doc_data, mask

    def set_train_count_data(self):
        """Set np.ndarray of bag of words for train set; rows: docs, cols: word counts."""
        self.train_count_data = self.preprocess_data_to_array(subset='train')
        if self.remove_zero_vocab_docs:
            self.train_count_data, mask = self.remove_zero_count_docs(doc_count_data=self.train_count_data)
            original_train_size = len(self.train_label_vals)
            self.train_label_vals = self.train_label_vals[mask]
            print('After zero count doc removal:\n Kept %d samples from original %d train'
                  % (len(self.train_label_vals), original_train_size))
        return None

    def set_test_count_data(self):
        """Set np.ndarray of bag of words for test set; rows: docs, cols: word counts."""
        self.test_count_data = self.preprocess_data_to_array(subset='test')
        self.test_count_data = self.remove_zero_count_docs(doc_count_data=self.train_test_data)
        return None

    def get_doc_lens(self):
        """Compute constant doc length via max len of train set."""
        self.train_doc_lens = np.sum(self.train_count_data, axis=self.doc_axis)
        self.max_doc_len = np.max(self.train_doc_lens)
        self.med_doc_len = np.median(self.train_doc_lens)
        return None

    @staticmethod
    def stats_nonzero_word_count_per_doc(word_count_data: np.ndarray, doc_axis: int = 1) -> Tuple[float, float]:
        """Stats for number of nonzero word counts per document."""
        nonzero_word_counts = np.sum(word_count_data.astype(np.bool_), axis=doc_axis)
        return np.median(nonzero_word_counts), np.mean(nonzero_word_counts)

    def make_uniform_doc_lens(self, word_count_data: np.ndarray, doc_axis: int = 1,
                              strategy: str = 'median') -> np.ndarray:
        """For each doc sum of each doc -> constant
        This constant is determined by the train dataset.
        np.ndarray -> np.ndarray
        """
        if strategy == 'median':
            static_doc_len = self.med_doc_len
        elif strategy == 'max':
            static_doc_len = self.max_doc_len
        else:
            raise Exception('Static doc len strategy %s not implemented' % strategy)
        reshaped_sums = np.sum(word_count_data, axis=doc_axis).reshape(len(word_count_data), 1)
        scaled_word_count_data = (static_doc_len / reshaped_sums) * word_count_data
        return scaled_word_count_data
