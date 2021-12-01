"""
EM train and test, after EDA.

Summary
-------
Follows Nigam et al "Semi-Supervised Text Classification Using EM" [2006, Ch 3 in SSL Book].

Treat each document as a vector of word counts (cf. preprocessing.py).
Initialize a naive bayes classifier (NBC) on labeled data.
Use NBC to compute class membership probabilities on unlabeled data.
Define a loss function as a sum of labeled_loss + unlabeled_loss.
Apply EM to this loss function until lower bound is not improved.

Notes
-----
Each "Eq" referes to an equation in [Nigam] Section 3.2.

@author: ibarrien
@email: corps.des.nombres@gmail.com

"""

import pathlib
import pickle
import numpy as np


class EM_SSL(object):
    """ Expectation maximization, naive bayes + Dirichlet prior with \alpha_j = 2 (for all classes j)

        Params:
            labeled_count_data (np.ndarray): labeled data, typically a 10% subsample for SSL
            label_vals (np.ndarrray): label values for each labeled sample
            doc_axis (int): index for documents
            vocab_axis (int): index for word counts, i.e. vocab axis

        Notes:
            Input count data can be implemented as a count vectorizer on bag of words
            See preprocessing.py for an example
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
        labels_list = list(self.label_set)
        labels_list.sort(reverse=False)
        self.ordered_labels_list = labels_list
        self.n_labels = len(self.label_set)
        # Dynamic vals: defined and updated in EM
        self.count_data = self.labeled_count_data  # union w/ unlabeled after EM param initialization
        self.n_docs = len(self.count_data)  # initialize, later add unlabeled
        self.class_mask = np.array([])
        self.this_class_count_data = np.array([])
        self.n_docs_in_class: float = 0  # given class j, num docs in class (float if unlabeled data)
        self.theta_j: float = 0  # Naive Bayes class prob of class j
        # Hash maps of statistics
        self.theta_j_per_class = np.zeros(self.n_labels)  # vals = theta_j: float via Eq 3.6 in SSL
        self.theta_jt_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = theta_jt: np.ndarray (vocab_size,)
        self.n_docs_per_class = np.zeros(self.n_labels)   # vals = num docs: int
        self.total_word_count_per_class = np.zeros(self.n_labels)  # vals: total words: int
        self.word_counts_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = word_counts: np.ndarray (vocab_size, )

    def set_in_class_mask(self, class_idx: int = 0):
        """Data mask of class label."""
        self.class_mask = self.label_vals == class_idx

    def set_count_data_in_class(self):
        """Set word count of labeled data subset corresponding to a class."""
        self.this_class_count_data = self.count_data[self.class_mask]

    def compute_doc_counts_in_class(self, only_labeled_data=True) -> float:
        """Compute n_j: num (potentially fractional) documents with class label. Assumes class mask is set."""
        if only_labeled_data:
            n_docs_in_class = np.sum(self.class_mask, axis=0)
        return n_docs_in_class

    def compute_theta_j(self, n_docs_in_class: float, only_labeled_data=True) -> float:
        """Compute single class proba = P(class = j | theta) via Eq 3.6. Assumes class mask is set."""
        if only_labeled_data:
            theta_j = (n_docs_in_class + 1) / (self.n_docs + self.n_labels)
        return theta_j

    def compute_word_counts_in_class(self, class_count_data: np.ndarray, only_labeled_data: bool = True) -> np.ndarray:
        """For given class j (implicit), compute each words' count.

        Params:
            class_count_data (np.ndarray): word count_data in given a fixed class
            only_labeled_data (bool): whehter all count data is labeled

        Returns:
            n_jt (np.ndarray): vector of words' count in class_count_data
            n_jt.shape = (vocab_size,)
        """
        if only_labeled_data:
            n_jt_vect = np.sum(class_count_data, axis=self.doc_axis)

        return n_jt_vect

    @staticmethod
    def compute_total_words(word_count_data: np.ndarray) -> float:
        """Compute total amount of words, counting multiplicities (i.e. full frequency sum)."""
        return np.sum(word_count_data)

    def compute_theta_vocab_j(self, word_counts_j: np.ndarray, total_word_count_j: float) -> np.ndarray:
        """For each word t and fixed class j, compute theta_tj values via Eq. 3.5:
            theta_tj = (1 + count_of_word_t_in_class_j) / (vocab_size + total_sum_words_class_j)

        Returns:
            theta_j_vocab: shape(words_counts_j) = (vocab_size, )
        """
        theta_j_vocab = (1 + word_counts_j) / (self.vocab_size + total_word_count_j)  # Eq 3.5
        return theta_j_vocab

    def compute_all_thetas_for_labeled_data(self):
        """For each class j, labeled docs, and words, compute the "mixture" theta:
            [theta_j] and  [theta_jt: t in words = vocab]
        """
        for j in self.ordered_labels_list:
            self.set_in_class_mask(class_idx=j)
            self.set_count_data_in_class()  # -> self.this_class_count_data
            n_docs_in_class = self.compute_doc_counts_in_class()
            self.n_docs_per_class[j] = n_docs_in_class  # {N_j}_j, only updated in EM
            theta_j = self.compute_theta_j(n_docs_in_class=n_docs_in_class)  # single class proba, not indexed by words
            self.theta_j_per_class[j] = theta_j
            word_counts_j = self.compute_word_counts_in_class(class_count_data=self.this_class_count_data)
            self.word_counts_per_class[j] = word_counts_j  # {N_jt: t in vocab}_j, only update in EM
            total_word_count_j = self.compute_total_words(word_count_data=word_counts_j)
            self.total_word_count_per_class[j] = total_word_count_j
            theta_j_vocab = self.compute_theta_vocab_j(word_counts_j=word_counts_j,
                                                       total_word_count_j=total_word_count_j)
            self.theta_jt_per_class[j] = theta_j_vocab

    def compute_doc_proba_per_class(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i, for each class j compute P(x = x_i | class = j; theta) via Eq 3.2.

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size,)
        """
        doc_probas_per_class = np.zeros(self.n_labels)
        for j in self.ordered_labels_list:
            doc_proba_j = np.prod(self.theta_jt_per_class[j] ** doc_word_counts, axis=0)
            doc_probas_per_class[j] = doc_proba_j
        return doc_probas_per_class

    def compute_unnormalized_class_probas_doc(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i, for each class j compute unnorm_P(c = j | x = x_i; theta)

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size,)

        Notes:
            Returns unnormalized factor, i.e. numerator of Eq 3.7 in SSL
            Used directly, without normalizing, to compute loss: Eq 3.8 in SSL
        """
        doc_probas_per_class = self.compute_doc_proba_per_class(doc_word_counts=doc_word_counts)
        u_class_probas_doc = np.prod([self.theta_j_per_class, doc_probas_per_class], axis=0)
        return u_class_probas_doc

    def compute_normalized_class_probas_doc(self, doc_word_counts: np.ndarray):
        """For fixed doc x_i and theta, and each j: compute P(c = j | x_i; theta) via Eq 3.7 in SSL.

        Notes:
            Main 'inference' method.
            Used to compute class probas of unlabeled docs -> EM updates
        """
        unnormalized_class_probas = self.compute_unnormalized_class_probas_doc(doc_word_counts=doc_word_counts)
        denom = np.sum(unnormalized_class_probas)
        normalized_class_probs = unnormalized_class_probas / denom
        return normalized_class_probs

    def E_step(self, initial_step=False):
        if initial_step:
            # supervised: count data is all labeled
            assert self.count_data.shape == self.labeled_count_data.shape, "First E_step is not all labeled data"
            self.compute_all_thetas_for_labeled_data()

        else:
            # supervised + unsupervised
            raise Exception("not implemented")


"""DRIVER FOR TESTING PURPOSES ONLY """
# PATHS
version = 'ten_pct_train_no_english_filter'
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_data_input_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v_%s.pkl' % version)
train_labels_input_filepath = pathlib.PurePath(main_dir, 'train_labels_v_%s.pkl' % version)
# train_count_vectorizer_filepath = pathlib.PurePath(main_dir, 'train_count_vectorizer_v_%s' % version)
train_count_data = pickle.load(open(train_data_input_filepath, 'rb'))
train_label_vals = pickle.load(open(train_labels_input_filepath, 'rb'))

# RUN EM (for now only sets up supervised data params)
em = EM_SSL(labeled_count_data=train_count_data,
            label_vals=train_label_vals)
em.E_step(initial_step=True)


# TODO: add M_step with unlabeled data

# Basic assertion tests
assert np.isclose(a=np.sum(em.theta_j_per_class), b=1.0, atol=1e-5), "theta_j's should sum to 1"
assert em.word_counts_per_class.shape == (em.n_labels, em.vocab_size), "word counts per class has wrong shape"
# Infernce example
t = train_count_data[0]
class_probas = em.compute_normalized_class_probas_doc(doc_word_counts=t)
assert np.isclose(a=np.sum(class_probas), b=1.0, atol=1e-5), "inference class probas should sum to 1"
print("Congrats, all assertions passed.")
