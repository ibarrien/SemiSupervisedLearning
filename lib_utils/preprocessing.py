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


def _stem(text: str, stemmer, min_len_stemmed: int = 2) -> str:
    """Remove stemming from sentence"""
    new_text = ' '.join([stemmer.stem(x) for x in text.split()])
    new_text = ' '.join([x for x in new_text.split() if len(x) > min_len_stemmed])
    return new_text


def is_english(text: str, english_vocab: List[str]) -> str:
    """Remove words not in english vocab."""
    english_words_text = ' '.join([x for x in text.split() if x in english_vocab])
    return english_words_text


def _process_text(document_as_single_str: str, tokens_to_remove: List[str] = None,
                 english_vocab: List[str] = None, stemmer=None) -> str:
    """Basic text processing on a single string."""
    # filtered.translate(str.maketrans('', '', string.punctuation))
    filtered = document_as_single_str.lower()
    filtered = re.sub('[^a-zA-Z]', ' ', filtered)
    filtered = re.sub(r'\[[0-9]*\]', ' ', filtered)
    filtered = re.sub(r'\s+', ' ', filtered)  # white space chars
    if stemmer:
        filtered = _stem(filtered, stemmer=stemmer)
    if tokens_to_remove:
        filtered = remove_stop_words(filtered, tokens_to_remove=tokens_to_remove)
    if english_vocab:
        filtered = is_english(filtered, english_vocab=english_vocab)
    return filtered


class TextPreProcessor:
    """Preprocessing of text documents.

    Params:
        english_vocab (List(str) or None): English words to restrict data to
        vocab_axis (int): index of words in vocabulary
        remove_fields (Tuple(str)): fields to remove from 20 Newsgroup

    Notes:
        labeled_train_data_sample: sample from full train data (can vary in experiments)
        unlabeled_data: typically fixed over experiments
        test_data: fixed over experiments

        To avoid overfitting on 20NewsGroups, recommended to set:
        remove_fields=('headers', 'footers', 'quotes')
    """
    def __init__(self, tokens_to_remove, english_vocab=None,
                 vocab_axis: int = 1,
                 label_names_key: str = 'target_names',
                 label_vals_key: str = 'target',
                 remove_zero_vocab_docs: bool = True,
                 n_labeled_train_samples=20,
                 n_unlabeled_train_samples=1000,
                 remove_fields=('headers', 'footers', 'quotes')):
        self.tokens_to_remove = tokens_to_remove
        self.english_vocab = english_vocab
        self.vocab_axis = vocab_axis
        self.remove_fields = remove_fields
        self.port_stemmer = None  # PorterStemmer()
        self.label_names_key = label_names_key
        self.label_vals_key = label_vals_key
        self.remove_zero_vocab_docs = remove_zero_vocab_docs
        self.n_unlabeled_train_samples = n_unlabeled_train_samples
        self.n_labeled_train_samples = n_labeled_train_samples
        self.labeled_train_data_sample = np.array([])  # raw train data sample
        self.labeled_train_sample_count_data = np.array([])  # count vectorizer of labeled train data
        self.train_sample_label_vals = np.array([])  # label vals of training data
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
                                        shuffle=True,
                                        random_state=1,
                                        remove=self.remove_fields)

        return data_bunch

    def set_static_full_train_raw_data(self):
        """Fetch raw train data; this method should only be called once."""
        # TODO: make optional if full train is already downloaded/cached
        self.full_train_bunch = self.load_data(subset='train')  # 11,314  total samples
        self.full_train_label_names = self.full_train_bunch[self.label_names_key]  # List[str], len = 20
        self.full_train_data = np.array(self.full_train_bunch.data)  # np.ndarray[str]
        print('full train raw data shape:', self.full_train_data.shape)
        self.full_train_label_vals = np.array(self.full_train_bunch[self.label_vals_key])  # List[str], len = len(self.train_data)
        print('min, max full train data label vals = %d, %d'
              % (min(self.full_train_label_vals), max(self.full_train_label_vals)))

    def set_static_raw_unlabeled_data(self):
        """Fix sample of 'unlabaled' data from full train."""
        self.unlabeled_train_indices = np.random.choice(a=range(len(self.full_train_data)),
                                                        size=self.n_unlabeled_train_samples,
                                                        replace=False)
        self.unlabeled_train_data = np.array(self.full_train_data[self.unlabeled_train_indices])
        print('unlabeled train data shape:', self.unlabeled_train_data.shape)
        self.avail_train_indices = list(set(range(len(self.full_train_data))) - set(self.unlabeled_train_indices))
        print('num avail train indices complement to unlabeled: %d' % len(self.avail_train_indices))
        if not self.avail_train_indices:
            raise Exception("Sampling zero labeled train examples; decrease unlabeled sample size.")

    def set_n_labeled_train_samples(self, n: int) -> None:
        """Set num (rand uniform) train samples to select."""
        self.n_labeled_train_samples = n

    def set_sample_raw_train_data(self):
        """Select sample of train data from complement of unlabeled data.

        Notes:
            run once per eval iteration on varying train size."
        """
        assert len(self.unlabeled_train_data) > 0, "Set unlabeled data before setting current labeled train sample."
        print('num train samples to select: %d' % self.n_labeled_train_samples)
        self.labeled_train_sample_indices = np.random.choice(a=self.avail_train_indices,
                                                      size=self.n_labeled_train_samples,
                                                      replace=False)


        self.labeled_train_data_sample = self.full_train_data[self.labeled_train_sample_indices]
        print('at set_sample_raw_train_data')
        print('size:', self.labeled_train_data_sample.shape)

        self.train_sample_label_vals = self.full_train_label_vals[self.labeled_train_sample_indices]

    def set_static_raw_test_data(self):
        """Fetch and fix standard 20 NewsGroup Test Data; should only be called once"""
        full_test_bunch = self.load_data(subset='test')
        self.full_test_data = np.array(full_test_bunch.data)  # List[str]
        self.full_test_label_vals = np.array(full_test_bunch[self.label_vals_key])

    def process_documents_text(self, documents_array: np.ndarray) -> List[str]:
        # Apply basic text pre-processing to loaded data.
        # print('received text_list:', text_list.shape, len(text_list))
        assert len(documents_array) > 0, "Received no documents for text preprocessing"
        if type(documents_array) == str or type(documents_array) == np.str_:
            print("Warning in process_documents_text: "
                  "received single doc as str, not array; converting to array")
            documents_array = np.array([documents_array])
        x_proc = [_process_text(document_as_single_str=doc,
                                tokens_to_remove=self.tokens_to_remove,
                                english_vocab=self.english_vocab)
                  for doc in documents_array]

        return x_proc

    def preprocess_data_to_array(self, subset: str = 'train') -> np.ndarray:
        """Preprocess text to 2-d array of word counts via count vectorization.

        Params
        ------
        subset (str): type of dataset wrt building generative model or inference

        Returns
        -------
        data_vect_array (np.ndarray): representation of documents as word counts

        Notes:
        data_vect_array: Rows = document index; Columns = word count index
        """
        data = np.array([])
        if subset == 'train':
            # use count vectorizer to fit and transform on labeled train data
            self.count_vectorizer = CountVectorizer()

            if len(self.labeled_train_data_sample) == 0:
                raise Exception("(raw) train data not set; run set_train_raw_data()")
            data = self.labeled_train_data_sample
            x_proc = self.process_documents_text(documents_array=data)
            x_proc_vect = self.count_vectorizer.fit_transform(x_proc)  # scipy.sparse.csr.csr_matrix
            self.vocab = self.count_vectorizer.get_feature_names()  # List[str]
        else:
            assert self.count_vectorizer, "Count vectorizer not set, run subset='train' first"
            # use count_vectorizer transform only, do not fit
            if subset == 'unlabeled':
                data = self.unlabeled_train_data
                print('got data=unlabeled, shape=', data.shape)
            elif subset == 'test':
                data = self.full_test_data
            else:
                raise Exception("Preprocessing data type must be 'train', 'unlabeled', or 'test'")
            if len(data) == 0:
                raise Exception("(raw) %s data not set; run\n set_test_raw_data()" % subset)
            # print('data shape = ', data.shape)
            # print('data type = ', type(data))
            x_proc = self.process_documents_text(documents_array=data)
            x_proc_vect = self.count_vectorizer.transform(x_proc)
        data_vect_array = x_proc_vect.toarray()
        return data_vect_array

    @staticmethod
    def remove_zero_count_docs(doc_count_data: np.ndarray, vocab_axis: int = 1) -> Tuple[np.ndarray]:
        """Remove zero count doc vectors wrt *preprocessed vocab* and keep their labels.

        Notes: only applied labeled train and labeled test data, not unlabeled dataset.
        """
        count_sums = np.sum(doc_count_data, axis=vocab_axis)
        mask = count_sums > 0  # use original indices for label val retrieval
        nonzero_doc_data = doc_count_data[mask]
        return nonzero_doc_data, mask

    def set_labeled_train_sample_count_data(self):
        """Set np.ndarray of bag of words for (labeled) train set; rows: docs, cols: word counts.

        Notes:
            Since unlabeled data is random, this method does not apply to unlabeled dataset
        """
        self.labeled_train_sample_count_data = self.preprocess_data_to_array(subset="train")
        print('labeled train sample count data before zero doc removal shape',
              self.labeled_train_sample_count_data.shape)
        if self.remove_zero_vocab_docs:
            print("Removing zero vocab docs from labeled train sample.")
            self.labeled_train_sample_count_data, mask =\
                self.remove_zero_count_docs(doc_count_data=self.labeled_train_sample_count_data)
            original_labeled_train_size = len(self.full_train_data)
            self.train_sample_label_vals = self.train_sample_label_vals[mask]
            print('After zero count doc removal:\n Kept %d samples from original %d train'
                  % (len(self.train_sample_label_vals), original_labeled_train_size))
        return None

    # TODO: set flag on whether to remove zero count (wrt labeled train) docs from unlabeled/test
    def set_unlabeled_count_data(self):
        """Set np.ndarray of bag of words for unlabeled set; rows: docs, cols: word counts."""
        self.unlabeled_count_data = self.preprocess_data_to_array(subset='unlabeled')
        # self.unlabeled_count_data, _mask = self.remove_zero_count_docs(doc_count_data=self.unlabeled_count_data)
        print('unlabeled count data shape', self.unlabeled_count_data.shape)
        return None

    def set_test_count_data(self):
        """Set np.ndarray of bag of words for test set; rows: docs, cols: word counts."""
        self.test_count_data = self.preprocess_data_to_array(subset='test')
        # self.test_count_data, _mask = self.remove_zero_count_docs(doc_count_data=self.test_count_data)
        print('test count data shape', self.test_count_data.shape)
        return None

    def get_train_doc_lengths(self, min_doc_len=20):
        """Compute constant doc length via max (or median) len of train set."""
        self.train_doc_lens = np.sum(self.labeled_train_sample_count_data, axis=self.vocab_axis)
        self.max_doc_len = max(min_doc_len, np.max(self.train_doc_lens))
        self.med_doc_len = max(min_doc_len, np.median(self.train_doc_lens))
        return None

    @staticmethod
    def stats_nonzero_word_count_per_doc(word_count_data: np.ndarray, vocab_axis: int = 1) -> Tuple[float, float]:
        """Stats for number of nonzero word counts per document."""
        nonzero_word_counts = np.sum(word_count_data.astype(np.bool_), axis=vocab_axis)
        return np.median(nonzero_word_counts), np.mean(nonzero_word_counts)

    def make_uniform_doc_lens(self, word_count_data: np.ndarray, vocab_axis: int = 1,
                              strategy: str = 'median') -> np.ndarray:
        """For each doc sum of each doc -> constant
        This constant is determined by the train dataset.
        np.ndarray -> np.ndarray
        """
        self.get_train_doc_lengths()
        if strategy == 'median':
            static_doc_len = self.med_doc_len
        elif strategy == 'max':
            static_doc_len = self.max_doc_len
        else:
            raise Exception('Static doc len strategy %s not implemented' % strategy)
        reshaped_sums = np.sum(word_count_data, axis=vocab_axis).reshape(len(word_count_data), 1)
        scaled_word_count_data = (static_doc_len / reshaped_sums) * word_count_data
        return scaled_word_count_data
