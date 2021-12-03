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
    """Expectation maximization on Semi supervised learning task

        Base classification model: naive bayes + Dirichlet prior with \alpha_j = 2 (for all classes j).
        Documents are represented as vectors of word counts aka "count data".

        Params:
            labeled_count_data (np.ndarray): labeled count data, typically a 10% subsample for SSL
            label_vals (np.ndarrray): label values for each labeled sample
            unlabeled_count_data (np.ndarray): count data without label vals
            doc_axis (int): index for documents
            vocab_axis (int): index for word counts, i.e. vocab axis

        Notes:
            Input count data can be implemented as a count vectorizer on bag of words
            See preprocessing.py for an example
    """
    def __init__(self, labeled_count_data: np.ndarray, label_vals: np.ndarray,
                 unlabeled_count_data: np.ndarray,
                 doc_axis: int = 0, vocab_axis: int = 1):

        # Static vals
        self.labeled_count_data = labeled_count_data  # (n_docs, n_words) LABELED COUNT DATA
        self.unlabeled_count_data = unlabeled_count_data
        self.n_unlabeled_docs = len(self.unlabeled_count_data)
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
        self.n_labeled_docs_per_class = {}  # populated only once, in initial E_step
        # Dynamic vals: defined and updated in EM
        self.curr_class_idx = 0  # current class label, used to subset data
        self.only_labeled_data = True  # whehter current data is fully labeled (False => leverage unlabeled)
        self.count_data = self.labeled_count_data  # union w/ unlabeled after EM param initialization
        self.n_docs = len(self.count_data)  # initialize, later add unlabeled
        self.class_mask = np.array([])
        self.this_class_count_data = np.array([])
        self.theta_j: float = 0  # Naive Bayes class prob of class j
        # Hash maps of statistics
        self.theta_j_per_class = np.zeros(self.n_labels)  # vals = theta_j: float
        self.theta_jt_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = theta_jt: np.ndarray
        self.n_docs_per_class = np.zeros(self.n_labels)   # vals = num docs: int
        self.total_word_count_per_class = np.zeros(self.n_labels)  # vals: total words: int
        self.labeled_word_counts_per_class = np.zeros([self.n_labels, self.vocab_size])
        self.word_counts_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = word_counts: np.ndarray
        self.unlabeled_this_class_probas = np.zeros(self.n_unlabeled_docs)  # for single j, each x_u: P(c=j | x_u)
        self.unlabeled_data_class_probas = np.array([self.n_unlabeled_docs, self.n_labels])  # [P(c=j | x_u)]

    def set_in_class_mask(self):
        """Data mask of class label."""
        self.class_mask = self.label_vals == self.curr_class_idx

    def set_this_class_count_data(self):
        """Set word count of labeled data subset corresponding to a class."""
        if self.only_labeled_data:
            # select labeled data in class
            self.this_class_count_data = self.count_data[self.class_mask]
        else:
            # need all unlabeled data since class membership is a probability
            self.this_class_count_data = self.unlabeled_count_data  # (n_docs, n_labels)

    def compute_doc_counts_in_class(self) -> float:
        """Compute n_j: num (potentially fractional) documents with class label. Assumes class mask is set."""
        if self.only_labeled_data:
            n_docs_in_class = np.sum(self.class_mask, axis=0)
            # Static (fixed) labeled docs class counts: computed only once
            self.n_labeled_docs_per_class[self.curr_class_idx] = n_docs_in_class
            # Dynamically updated
            self.n_docs_per_class[self.curr_class_idx] = n_docs_in_class  # {N_j}_j, updated in E_step
        else:
            self.unlabeled_this_class_probas = self.unlabeled_data_class_probas[:, self.curr_class_idx]
            fractional_docs_in_class = np.sum(self.unlabeled_this_class_probas)
            self.n_docs_per_class[self.curr_class_idx] = \
                self.n_labeled_docs_per_class[self.curr_class_idx] + fractional_docs_in_class

    def compute_theta_j(self):
        """Compute single class proba = P(class = j | theta) via Eq 3.6. Assumes class mask is set."""
        # theta_j = 0.0
        n_docs_in_class = self.n_docs_per_class[self.curr_class_idx]
        theta_j = (n_docs_in_class + 1) / (self.n_docs + self.n_labels)
        self.theta_j_per_class[self.curr_class_idx] = theta_j

    def compute_word_counts_in_class(self):
        """For given class j (implicit), compute each words' count.

        Params:
            class_count_data (np.ndarray): word count_data in fixed class if labeled, else all word count_data
            only_labeled_data (bool): whehter all count data is labeled

        Computes:
            n_jt (np.ndarray): vector of words' count in class_count_data
            n_jt.shape = (vocab_size,)
        """
        if self.only_labeled_data:
            n_jt_vect = np.sum(self.this_class_count_data, axis=self.doc_axis)
            self.word_counts_per_class[self.curr_class_idx] = n_jt_vect  # (vocab_size,)
            self.labeled_word_counts_per_class[self.curr_class_idx] = n_jt_vect  # fix a copy
        else:
            # For this class j, and each x_u: compute P(c = j|x_u, theta) * x_u
            class_scaled_count_data = np.multiply(self.unlabeled_this_class_probas, self.unlabeled_count_data.T).T
            # Compute prob scaled word counts across all unlabeled docs
            unlabled_n_jt_vect = np.sum(class_scaled_count_data, axis=self.doc_axis)  # (vocab_size,)
            self.word_counts_per_class[self.curr_class_idx] = \
                self.labeled_word_counts_per_class[self.curr_class_idx] + unlabled_n_jt_vect  # (vocab_size,)

    @staticmethod
    def compute_total_words(word_count_data: np.ndarray) -> float:
        """Compute total amount of words, counting multiplicities (i.e. full frequency sum)."""
        return np.sum(word_count_data)

    def compute_total_words_in_class(self):
        """Compute total (potentially fractional) total words in current class."""
        # word_counts_per_class is already labeled_data/ +unlabeled aware
        self.total_word_count_per_class[self.curr_class_idx] = np.sum(self.word_counts_per_class[self.curr_class_idx])

    def compute_theta_vocab_j(self):
        """For each word t and fixed class j, compute theta_tj values via Eq. 3.5:
            theta_tj = (1 + count_of_word_t_in_class_j) / (vocab_size + total_sum_words_class_j)

        Computes:
            theta_j_vocab: shape(words_counts_j) = (vocab_size, )
        """
        word_counts_j = self.word_counts_per_class[self.curr_class_idx]
        total_word_count_j = self.total_word_count_per_class[self.curr_class_idx]
        theta_j_vocab = (1 + word_counts_j) / (self.vocab_size + total_word_count_j)  # Eq 3.5
        self.theta_jt_per_class[self.curr_class_idx] = theta_j_vocab

    def compute_all_thetas(self):
        """For each class j, labeled docs, and words, compute the "mixture" theta:
            [theta_j] and  [theta_jt: t in words = vocab]
        These are the maximum a posteriori (MAP) estimates of the Naive Bayes model.
        """
        for j in self.ordered_labels_list:
            self.curr_class_idx = j
            if self.only_labeled_data:
                self.set_in_class_mask()
            # else: leverage labeled_data + unlabeled_data
            self.set_this_class_count_data()  # if unlabeled, then leverage all unlabeled data
            self.compute_doc_counts_in_class()  # -> self.n_docs_per_class
            self.compute_theta_j()  # -> self.theta_j_per_class
            self.compute_word_counts_in_class()  # -> self.word_counts_per_class
            self.compute_total_words_in_class()  # -> self.total_word_count_per_class
            self.compute_theta_vocab_j()

    def compute_doc_proba_per_class(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i, for each class j compute P(x = x_i | class = j; theta) via Eq 3.2.

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size,)

        Example:

        """
        doc_probas_per_class = np.array([np.prod(self.theta_jt_per_class[j] ** doc_word_counts, axis=0)
                                         for j in self.ordered_labels_list])

        return doc_probas_per_class

    def compute_unnormalized_class_probas_doc(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i, for each class j compute unnorm_P(c = j | x = x_i; theta)

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size, )

        Computes:
            P(c = j | theta) * P(x_i | c = j; theta), shape = (n_labels, )

        Notes:
            Returns unnormalized factor, i.e. numerator of Eq 3.7 in SSL
            Used directly, without normalizing, to compute loss: Eq 3.8 in SSL
        """
        doc_probas_per_class = self.compute_doc_proba_per_class(doc_word_counts=doc_word_counts)
        # u_class_probas_doc = np.prod([self.theta_j_per_class, doc_probas_per_class], axis=0)
        u_class_probas_doc = np.multiply(self.theta_j_per_class, doc_probas_per_class)
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

    def compute_class_probas_unlabeled_data(self):
        """For each unlabeled doc x_u and class j, compute P(c = j | x_u, theta).

        Notes:
            shape(unlabeled_data_class_probas) = (n_unlabeled_docs, n_labels)
        """
        # np.array([self.compute_normalized_class_probas_doc(u) for u in self.unlabeled_count_data])
        self.unlabeled_data_class_probas = np.apply_along_axis(func1d=em.compute_normalized_class_probas_doc,
                                                               axis=self.vocab_axis,
                                                               arr=self.unlabeled_count_data)

    def E_step(self):
        """Estimate expectations, given current model params.

        Computes:
            For each unlabeled doc and current theta, compute class probas.
        """
        self.compute_class_probas_unlabeled_data()  # self.unlabeled_data_class_probas

    def M_step(self):
        """Maximize likelihood of model params using current expecations

        Computes: Re-estimate of theta using (fractional) unlabeled class probas.
        """
        self.compute_all_thetas()

    def initialize_EM(self):
        """Initial computations prior to expecation-maximization loop."""
        self.only_labeled_data = True
        assert self.count_data.shape == self.labeled_count_data.shape, "First E_step is not all labeled data"
        self.M_step()  # Builds the initial NBC thetas from labeled docs only
        self.only_labeled_data = False
        # update total number of documents being leveraged
        self.n_docs = len(self.labeled_count_data) + len(self.unlabeled_count_data)

    def compute_labeled_loss(self) -> float:
        """Compute loss attributed to labeled data.

        sum_{x labeled} log (P(class(x) | theta).P(x | c = class(x), theta
        """
        joint_probas = np.apply_along_axis(func1d=self.compute_unnormalized_class_probas_doc,
                                           axis=self.vocab_axis,
                                           arr=self.labeled_count_data)
        # joint_probas shape = (n_train_count, n_labels)
        self.joint_probas_of_label = np.array([joint_probas[k][self.label_vals[k]]
                                              for k in range(len(joint_probas))])

        loss = np.sum(np.log(self.joint_probas_of_label))  # sum across all labeled docs

        return loss

    def compute_unlabeled_loss(self) -> float:
        joint_probas = np.apply_along_axis(func1d=self.compute_unnormalized_class_probas_doc,
                                         axis=self.vocab_axis,
                                         arr=self.unlabeled_count_data)
        # joint_probas_per_class shape = (n_train_count, n_labels)
        joint_probas_across_classes = np.sum(joint_probas, axis=1)
        loss = np.sum(np.log(joint_probas_across_classes))  # sum across all unlabeled docs
        return loss

    def compute_total_loss(self) -> float:
        """Compute - (log(P(theta)) + loss(labeled_data) + loss(unlabeled_data))

        Returns:
            Total log loss >= 0.

        Notes:
            "Our prior distribution is formed with the product of Dirichlet distributions: one
            for each class multinomial and one for the overall class probabilities

        ToDo: compute P(theta) = prior distribution over (all?) paremeters
        """
        total_loss = self.compute_labeled_loss() + self.compute_unlabeled_loss()
        return -total_loss

    def run_EM_loop(self, max_n_iter=5):
        """Run expectation maximization until delta convergence or max iters."""
        self.initialize_EM()
        for _ in range(max_n_iter):
            # TODO: add delta improvement criterion for early stopping
            curr_loss = self.compute_total_loss()
            print('curr loss: %0.2f' % curr_loss)
            self.E_step()
            self.M_step()


"""DRIVER FOR TESTING PURPOSES ONLY """
# PATHS
version = 'ten_pct_train_no_english_filter'
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_data_input_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v_%s.pkl' % version)
train_labels_input_filepath = pathlib.PurePath(main_dir, 'train_labels_v_%s.pkl' % version)

# Data Load and Set
train_count_data = pickle.load(open(train_data_input_filepath, 'rb'))
train_label_vals = pickle.load(open(train_labels_input_filepath, 'rb'))
train_labeled_pct = 0.8
n_labeled_samples = int(train_labeled_pct * len(train_count_data))
labeled_count_data = train_count_data[: n_labeled_samples]
train_label_vals = train_label_vals[: n_labeled_samples]
unlabeled_count_data = train_count_data[n_labeled_samples:]

# RUN EM
em = EM_SSL(labeled_count_data=labeled_count_data,
            label_vals=train_label_vals,
            unlabeled_count_data=unlabeled_count_data)

em.run_EM_loop()

# ASSERTION TESTS (todo: write a seperate unit_test.py)
assert em.word_counts_per_class.shape == (em.n_labels, em.vocab_size), "word counts per class has wrong shape"
assert np.isclose(a=np.sum(em.theta_j_per_class), b=1.0, atol=1e-5), "theta_j's should sum to 1"

# Check computed class probas on unlabeled data
for u_idx in range(len(em.unlabeled_count_data)):
    assert np.isclose(a=np.sum(em.unlabeled_data_class_probas[u_idx]), b=1.0, atol=1e-5), \
        "u_doc %d class probs should sum 1" % u_idx


print("Congrats, all assertions passed.")

