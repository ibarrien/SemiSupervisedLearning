"""
EM train and test, after EDA.

@author: ivbarrie
"""

import pathlib
import pickle
import numpy as np

class EM(object):
    """ Expectation maximization, naive bayes + Dirichlet prior
    """
    def __init__(self, labeled_count_data: np.ndarray, label_vals: np.ndarray,
                 unlabeled_count_data: np.ndarray = None,
                 doc_axis: int = 0, vocab_axis: int = 1):
        # Static vals
        self.labeled_count_data = labeled_count_data  # (n_docs, n_words) LABELED COUNT DATA
        self.unlabeled_count_data = unlabeled_count_data
        self.vocab_axis = vocab_axis
        self.doc_axis = doc_axis
        self.label_vals = label_vals
        self.vocab_size = np.shape(self.labeled_count_data)[self.vocab_axis]
        assert len(self.labeled_count_data) == len(label_vals)
        self.label_set = set(np.unique(label_vals))
        self.n_labels = len(self.label_set)
        # Dynamic vals: defined and updated in EM
        self.count_data = self.labeled_count_data  # union w/ unlabeled after EM param initialization
        self.n_docs = len(self.count_data)  # initialize, later add unlabeled
        self.word_counts_per_class = np.array([])
        self.class_mask = np.array([])
        self.this_class_count_data = np.array([])
        self.n_docs_in_class: float = 0  # given class j, num docs in class (float if unlabeled data)
        self.theta_j: float = 0  # Naive Bayes class prob of class j

    def set_in_class_mask(self, class_idx: int = 0):
        """Data mask of class label."""
        self.class_mask = self.label_vals == class_idx

    def set_count_data_in_class(self):
        """Set word count of labeled data subset corresponding to a class."""
        self.this_class_count_data = self.count_data[self.class_mask]

    def compute_doc_counts_in_class(self):
        """Compute n_j: num (potentially fractional) documents with class label."""
        self.n_docs_in_class = np.sum(self.class_mask, axis=0)

    def compute_class_proba(self):
        """Compute P(class = j | theta_j). Assumes class mask is set"""
        self.theta_j = (self.n_docs_in_class + 1) / (self.n_docs + self.n_labels)  # float

    def compute_doc_proba_in_class(self, class_idx: int) -> np.ndarray:
        """For fixed j, and all docs x_i compute array:
            P(c = j | theta_j) * P(x = x_i | class = j, theta)

        Returns:
            doc_probas_unnormalized: shape = (n_docs_in_class
        """
        # print('setting class = %d' % class_idx)
        self.set_in_class_mask(class_idx=class_idx)
        self.set_count_data_in_class()
        self.compute_doc_counts_in_class()  # N_j: int if labeled, float if +unlabeled
        self.compute_class_proba()  # theta_j
        n_jt_vect = np.sum(self.this_class_count_data, axis=self.doc_axis)  # (vocab_size,): count of words in class j
        theta_jt_vect = (n_jt_vect + 1) / (self.n_docs_in_class + self.vocab_size)  # (vocab_size,)
        theta_jt_scaled = theta_jt_vect ** self.this_class_count_data
        # doc_probas_unnormalized.shape = (n_docs_in_class, )
        doc_probas_unnormalized = self.theta_j * np.prod(theta_jt_scaled, axis=self.vocab_axis)
        return doc_probas_unnormalized

    def compute_conditional_class_probas(self):
        """For all docs x_i and classes j compute
            P(c = j | x = x_i, theta)
        """
        self.doc_class_probas = [self.compute_doc_proba_in_class(class_idx=j)
                                 for j in self.label_set]
        # sum([len(X[k]) for k in range(len(X))]) = len(self.labeled_count_data)
        # TODO: self.normalization_vals =



    def E_step(self, initial_step=False):
        if initial_step:
            # supervised: count data is all labeled
            assert self.count_data.shape == self.labeled_count_data, "First E_step is not all labeled data"
            self.compute_conditional_class_probas()
        else:
            # supervised + unsupervised
            raise Exception("not implemented")

    # word-k count in class j: mask_j[:, k]

# DRIVER
# PATHS
version = 'ten_pct_train_no_english_filter'
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_data_input_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v_%s.pkl' % version)
train_labels_input_filepath = pathlib.PurePath(main_dir, 'train_labels_v_%s.pkl' % version)
# train_count_vectorizer_filepath = pathlib.PurePath(main_dir, 'train_count_vectorizer_v_%s' % version)
train_count_data = pickle.load(open(train_data_input_filepath, 'rb'))
train_label_vals = pickle.load(open(train_labels_input_filepath, 'rb'))

em = EM(labeled_count_data=train_count_data,
        label_vals=train_label_vals)

em.compute_conditional_class_probas()