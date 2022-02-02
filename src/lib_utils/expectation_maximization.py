"""
EM train and test, after data preprocessing and train/test splits.

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

from typing import Dict
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
            max_em_iters (int): max num EM iterations
            min_em_loss_delta (float): min improvement in non-negative loss during EM iterations

        Notes:
            Input count data can be implemented as a count vectorizer on bag of words
            See preprocessing.py for an example
    """
    def __init__(self, labeled_count_data: np.ndarray, label_vals: np.ndarray,
                 unlabeled_count_data: np.ndarray,
                 test_count_data: np.ndarray = None,
                 test_label_vals: np.ndarray = None,
                 doc_axis: int = 0, vocab_axis: int = 1,
                 max_em_iters: int = 20, min_em_loss_delta: float = 1e-2):

        # Static vals
        self.labeled_count_data = labeled_count_data  # (n_docs, n_words) LABELED COUNT DATA
        self.unlabeled_count_data = unlabeled_count_data
        self.n_unlabeled_docs = len(self.unlabeled_count_data)
        self.vocab_axis = vocab_axis
        self.doc_axis = doc_axis
        self.label_vals = label_vals
        self.max_em_iters = max_em_iters
        self.min_em_loss_delta = min_em_loss_delta
        self.vocab_size = np.shape(self.labeled_count_data)[self.vocab_axis]
        assert len(self.labeled_count_data) == len(label_vals), \
            "Num labeled sample features = %d != num sample labels = %d" % \
            (len(self.labeled_count_data), len(label_vals))
        self.label_set = set(np.unique(label_vals))
        print(f' labeled train sample has {len(self.label_set)} unique labels')
        self.ordered_labels_list = list(range(20))
        self.n_labels = 20  # len(self.label_set)
        self.n_labeled_docs_per_class = np.zeros(self.n_labels)  # populated only once, in initial E_step
        # Dynamic vals: defined and updated in EM
        self.curr_class_idx = 0  # current class label, used to subset data
        self.only_labeled_data = True  # whehter current data is fully labeled (False => leverage unlabeled)
        self.count_data = self.labeled_count_data  # union w/ unlabeled after EM param initialization
        self.n_docs = len(self.count_data)  # initialize, later add unlabeled
        self.class_mask = np.array([])
        self.this_class_count_data = np.array([])
        self.theta_j: float = 0  # Naive Bayes class prob of class j
        self.preds = np.array([])  # prediction array for evaluation, e.g. on out-of-sample data
        # Hash maps of statistics
        self.theta_j_per_class = np.zeros(self.n_labels)  # vals = theta_j: float
        self.theta_j_vocab_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = theta_jt: np.ndarray
        self.n_docs_per_class = np.zeros(self.n_labels)   # vals = num docs: int
        self.total_word_count_per_class = np.zeros(self.n_labels)  # vals: total words: int
        self.labeled_word_counts_per_class = np.zeros([self.n_labels, self.vocab_size])
        self.word_counts_per_class = np.zeros([self.n_labels, self.vocab_size])  # vals = word_counts: np.ndarray
        self.unlabeled_this_class_probas = np.zeros(self.n_unlabeled_docs)  # for single j, each x_u: P(c=j | x_u)
        self.unlabeled_data_class_probas = np.zeros([self.n_unlabeled_docs, self.n_labels])  # [P(c=j | x_u)]
        self.test_accuracy_hist = dict()  # out-of-sample test hist, including model without unlabeled data
        self.total_em_iters = 0  # number of EM iters including unlabeled data
        self.test_count_data = test_count_data
        self.test_label_vals = test_label_vals

    def set_in_class_mask(self) -> None:
        """Data mask of class label."""
        self.class_mask = self.label_vals == self.curr_class_idx

        return None

    def set_this_class_count_data(self) -> None:
        """Set word count of labeled data subset corresponding to a class."""
        if self.only_labeled_data:
            # select labeled data in class
            self.this_class_count_data = self.count_data[self.class_mask]
        else:
            # need all unlabeled data since class membership is a probability
            self.this_class_count_data = self.unlabeled_count_data  # (n_docs, n_labels)

        return None

    def compute_doc_counts_in_class(self) -> None:
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

        return None

    def compute_theta_j(self) -> None:
        """Compute single class proba = P(class = j | theta) via Eq 3.6. Assumes class mask is set."""
        n_docs_in_class = self.n_docs_per_class[self.curr_class_idx]
        theta_j = (n_docs_in_class + 1) / (self.n_docs + self.n_labels)
        self.theta_j_per_class[self.curr_class_idx] = theta_j
        if np.isclose(theta_j, 0):
            print('WARNING: Got theta_j near zero for class=%d' % self.curr_class_idx)

        return None

    def compute_word_counts_in_class(self) -> None:
        """For given class j (implicit), compute each words' count; +fractional for unlabeled based on class probas.

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

        return None

    @staticmethod
    def compute_total_words(word_count_data: np.ndarray) -> float:
        """Compute total amount of words, counting multiplicities (i.e. full frequency sum)."""
        return np.sum(word_count_data)

    def compute_total_words_in_class(self) -> None:
        """Compute total (potentially fractional) total words in current class."""
        # word_counts_per_class is already labeled_data/ +unlabeled aware
        self.total_word_count_per_class[self.curr_class_idx] = np.sum(self.word_counts_per_class[self.curr_class_idx])

        return None

    def compute_theta_vocab_j(self) -> None:
        """For each word t and fixed class j, compute word probas per class:
            theta_tj values via Eq. 3.5:
            P(w_t | c_j; theta) = (1 + count_of_word_t_in_class_j) / (vocab_size + total_sum_words_class_j)

        Computes:
            theta_j_vocab: shape(words_counts_j) = (vocab_size, )

        Notes:
            Min(P(w_t | c_j)) = 1 / (vocab_size + total_sum_words_class_j)
            If word_t has sparse count, then theta_jt -> 0
            theta_j_vocab_per_class.shape = (n_classes, vocab_size)
        """
        word_counts_j = self.word_counts_per_class[self.curr_class_idx]  # (vocab_size, 1)
        total_word_count_j = self.total_word_count_per_class[self.curr_class_idx]  # int
        theta_j_vocab = (1 + word_counts_j) / (self.vocab_size + total_word_count_j)  # Eq 3.5
        self.theta_j_vocab_per_class[self.curr_class_idx] = theta_j_vocab  # (vocab_size, 1)

        return None

    def compute_all_thetas(self) -> None:
        """For each class j, labeled docs, and words, compute the "mixture" theta:
            [theta_j] and  [theta_jt: t in words = vocab]
        These are the maximum a posteriori (MAP) estimates of the Naive Bayes model.
        """
        for j in self.label_set:
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

        return None

    @staticmethod
    def compute_log_of_sums(log_factors: np.ndarray) -> np.ndarray:
        """Compute log of sums via LogExpSum trick (cf. Murphy 2012 Section 3.5.3)."""
        max_log = np.max(log_factors)
        summand = np.exp(log_factors - max_log)
        log_of_sums = np.log(np.sum(summand)) + max_log

        return log_of_sums

    def compute_unnormalized_class_log_probas_doc(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i, for each class j compute unnormalized log P(c = j | x = x_i; theta)

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size, )

        Returns:
            u_log_probas (np.ndarray): shape = (n_labels, )

        Notes:
            Returns unnormalized log probas, i.e. log(numerator) of Eq 3.7 in SSL
            Used directly, without normalizing, to compute loss: Eq 3.8 in SSL
        """

        u_log_probas = np.log(self.theta_j_per_class) + \
                       np.array([np.sum(doc_word_counts * np.log(self.theta_j_vocab_per_class[j]), axis=0)
                                for j in self.ordered_labels_list])

        return u_log_probas

    def compute_normalized_class_probas_doc(self, doc_word_counts: np.ndarray) -> np.ndarray:
        """For fixed doc x_i and theta, and each j: compute P(c = j | x_i; theta) via Eq 3.7 in SSL.

        Params:
            doc_word_counts: representation of a single document, shape = (vocab_size, )

        Returns:
            class_probas_normalized: class membership probabilities for the given doc x_i.

        Notes:
            Applies LogSumExp trick to compute normalization factor (i.e. denominator of Eq 3.7)
            Main 'inference' method, i.e. model predictions for class membership.
            Used to compute class probas of unlabeled docs during EM updates
        """
        unnormalized_class_log_probas = self.compute_unnormalized_class_log_probas_doc(doc_word_counts=doc_word_counts)
        log_of_sums = self.compute_log_of_sums(log_factors=unnormalized_class_log_probas)
        class_log_probas_normalized = unnormalized_class_log_probas - log_of_sums  # log(a/b) = log(a) - log(b)
        class_probas_normalized = np.exp(class_log_probas_normalized)

        return class_probas_normalized

    def compute_class_probas_unlabeled_data(self) -> None:
        """For each unlabeled doc x_u and class j, compute P(c = j | x_u, theta).

        Notes:
            shape(unlabeled_data_class_probas) = (n_unlabeled_docs, n_labels)
        """
        self.unlabeled_data_class_probas = np.apply_along_axis(func1d=self.compute_normalized_class_probas_doc,
                                                               axis=self.vocab_axis,
                                                               arr=self.unlabeled_count_data)

        return None

    def E_step(self) -> None:
        """Estimate expectations, given current model params.

        Computes:
            For each unlabeled doc and current theta, compute class probas.
        """
        self.compute_class_probas_unlabeled_data()  # self.unlabeled_data_class_probas

        return None

    def M_step(self) -> None:
        """Maximize likelihood of model params using current expecations

        Computes: Re-estimate of theta using (fractional) unlabeled class probas.
        """
        self.compute_all_thetas()

        return None

    def check_initial_M_step(self) -> None:
        total_class_counts = np.sum(self.n_labeled_docs_per_class, axis=0)
        n_labeled_train = self.labeled_count_data.shape[0]
        print(" Checking initial M step on only labeled train data...")
        assert total_class_counts == n_labeled_train, "Total class count = %d != %d labeled train samples" % \
            (total_class_counts, n_labeled_train)
        print(" Congrats, initial M step assertions passed.")

        return None

    def initialize_EM(self) -> None:
        """Initial computations prior to expecation-maximization loop."""
        self.only_labeled_data = True
        assert self.count_data.shape == self.labeled_count_data.shape, "First M_step is not on all labeled data"
        self.M_step()  # Builds the initial NBC thetas from labeled docs only
        self.check_initial_M_step()
        self.only_labeled_data = False
        # update total number of documents being leveraged
        self.n_docs = len(self.labeled_count_data) + len(self.unlabeled_count_data)

        return None

    def compute_prior_loss(self) -> float:
        """Compute log(P(theta)).

        Notes:
            Assume a factorized prior = P(theta_j).P(theta_tj)
            Recall Dirchlet priors each have \alpha = 2 => \alpha - 1 = 1
            Hence, take direct products of theta elements
        """
        # Compute log(P(theta_jt)) = log(prod_{i, j} theta_ij) = sum_{i,_j} (log(theta_ij)))
        theta_j_vocab_log = np.log(self.theta_j_vocab_per_class)  # shape = (n_classes, vocab_size)
        log_proba_theta_j_vocab = np.sum(np.sum(theta_j_vocab_log, axis=0))  # float

        # Compute log(P(theta_j))
        log_proba_theta_j = np.sum(np.log(self.theta_j_per_class))  # float

        return log_proba_theta_j + log_proba_theta_j_vocab

    def compute_labeled_loss(self) -> float:
        """Compute loss attributed to labeled data: sum(log(probas)).

        sum_{x labeled} log (P(class(x) | theta).P(x | c = class(x), theta
        """
        joint_log_factors = np.apply_along_axis(func1d=self.compute_unnormalized_class_log_probas_doc,
                                                axis=self.vocab_axis,
                                                arr=self.labeled_count_data)
        # joint_log_factors shape = (n_train_count, n_labels): unnormalized "joint"
        joint_log_factors_labeled_data = np.array([joint_log_factors[k][self.label_vals[k]]
                                                   for k in range(len(joint_log_factors))])

        loss = np.sum(joint_log_factors_labeled_data)  # sum across all labeled docs

        return loss

    def compute_unlabeled_loss(self) -> float:
        """Compute loss attributed to unlabeled data: sum(log(sum(probas)). """
        joint_log_factors = np.apply_along_axis(func1d=self.compute_unnormalized_class_log_probas_doc,
                                                axis=self.vocab_axis,
                                                arr=self.unlabeled_count_data)
        # joint_factors: (n_unlabeled_docs, n_labels)
        log_of_sums = np.apply_along_axis(func1d=self.compute_log_of_sums,
                                          axis=1,  # for each doc, sum across classes
                                          arr=joint_log_factors)
        # log_of_sums: (n_unlabeled_docs, )
        loss = np.sum(log_of_sums)  # sum across all unlabeled docs

        return loss

    def compute_total_loss(self) -> float:
        """Compute: -1 * (log(P(theta)) + log(P(labeled_data)) + log(P(unlabeled_data))

        Returns:
            Total log loss >= 0.

        Notes:
            "Our prior distribution is formed with the product of Dirichlet distributions: one
            for each class multinomial and one for the overall class probabilities."

        """
        total_loss = self.compute_prior_loss() + self.compute_labeled_loss() + self.compute_unlabeled_loss()

        return -total_loss

    def evaluate_on_data(self, count_data: np.array, label_vals: np.array) -> float:
        """Evaluate current model on a given set of documents.

        Params:
            count_data: array of documents, each doc represented by a word count vector
            label_vals: labels for count data

        Returns:
            Current model predictive (on the nose) accuracy
        """
        pred_probas = np.apply_along_axis(func1d=self.compute_normalized_class_probas_doc,
                                          axis=self.vocab_axis,
                                          arr=count_data)

        self.preds = np.argmax(pred_probas, axis=1)
        assert self.preds.shape == label_vals.shape, "Predictions have shape != ground truth vals."
        correct_preds = self.preds == label_vals
        n_correct = np.sum(correct_preds)
        pct_correct = n_correct / len(label_vals)

        return pct_correct

    def fit(self) -> None:
        """Run expectation maximization until delta convergence or max iters."""
        self.initialize_EM()
        # get test accuracy without using unlabeled data in training
        if self.test_count_data is not None and self.test_label_vals is not None:
            curr_test_acc = self.evaluate_on_data(count_data=self.test_count_data,
                                                  label_vals=self.test_label_vals)
            print(' out-of-sample test acc using only labeled data: %0.2f%%' % (100 * curr_test_acc))
            self.test_accuracy_hist[0] = curr_test_acc
        # start EM loop using unlabeled data as part of training
        curr_loss = np.inf
        for em_iter in range(self.max_em_iters):
            prev_loss = curr_loss
            curr_loss = self.compute_total_loss()
            print('  curr train loss: %0.2f' % curr_loss)
            self.E_step()
            self.M_step()
            self.total_em_iters += 1
            if self.test_count_data is not None and self.test_label_vals is not None:
                curr_test_acc = self.evaluate_on_data(count_data=self.test_count_data,
                                                      label_vals=self.test_label_vals)
                print('  curr out-of-sample test acc: %0.2f%%' % (100 * curr_test_acc))
                self.test_accuracy_hist[em_iter + 1] = curr_test_acc  # key 0 is for using only labeled data
            delta_improvement = prev_loss - curr_loss  # expect 0 <= curr_loss <= prev_loss
            if delta_improvement < self.min_em_loss_delta:
                print('Early stopping EM: delta improvement = %0.4f < min_delta = %0.4f'
                      % (delta_improvement, self.min_em_loss_delta))
                break

        return None

    def get_test_acc_hist(self) -> Dict[int, float]:
        """Get out-of-sample test accuracy history over EM iterations."""
        return self.test_accuracy_hist

    def only_labeled_test_acc(self) -> float:
        """Test accuracy using only labeled data."""
        return self.test_accuracy_hist[0]

    def last_em_iter_test_acc(self) -> float:
        """Test accuracy on last complete EM iteration that included unlabeled data."""
        assert self.total_em_iters > 0, "Did not train using unlabeled data."

        return self.test_accuracy_hist[self.total_em_iters]
