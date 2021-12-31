"""
Example driver for EM routine.
Assumes data has been preprocessed and stored locally.

@author: ibarrien
"""

import pickle
import pathlib
import numpy as np
from lib_utils.expectation_maximization import EM_SSL

# PATHS
version = 'ten_pct_train_no_english_filter'
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_data_input_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v_%s.pkl' % version)
train_labels_input_filepath = pathlib.PurePath(main_dir, 'train_labels_v_%s.pkl' % version)

# ORIGINAL Data Load and Set (n'est touche pas)
_count_data = pickle.load(open(train_data_input_filepath, 'rb'))
_label_vals = pickle.load(open(train_labels_input_filepath, 'rb'))

# SET LABELED DATA FOR EM
train_labeled_pct = 0.8  # remaining pct used as unlabeled in EM
n_labeled_samples = int(train_labeled_pct * len(_count_data))
train_label_vals = _label_vals[: n_labeled_samples]
labeled_count_data = _count_data[: n_labeled_samples]

# SET UNLABELED DATA FOR EM
unlabeled_count_data = _count_data[n_labeled_samples:]

# SET UNLABELED DATA LABELS FOR SANITY CHECKS
unlabeled_data_label_vals = _label_vals[n_labeled_samples:]

# RUN EM
em = EM_SSL(labeled_count_data=labeled_count_data,
            label_vals=train_label_vals,
            unlabeled_count_data=unlabeled_count_data,
            max_em_iters=10,
            min_em_loss_delta=2e-4)

em.fit()

# ASSERTION TESTS (todo: write a seperate unit_test.py)
assert em.word_counts_per_class.shape == (em.n_labels, em.vocab_size), "word counts per class has wrong shape"
assert np.isclose(a=np.sum(em.theta_j_per_class), b=1.0, atol=1e-5), "theta_j's should sum to 1"

# Check computed class probas on unlabeled data
for u_idx in range(len(em.unlabeled_count_data)):
    assert np.isclose(a=np.sum(em.unlabeled_data_class_probas[u_idx]), b=1.0, atol=1e-5), \
        "u_doc %d class probs should sum 1" % u_idx


print("Congrats, all assertions passed.")

# Evaluation on (used!) unlabeled data

# get naive baseline acc
vals, counts = np.unique(unlabeled_data_label_vals, return_counts=True)
max_freq = np.max(counts)  # num times the most frequent value appears
naive_acc = max_freq / len(unlabeled_count_data)
print('naive guess acc for unlabeled: %0.2f%%' % (100 * naive_acc))

pred_probas = np.apply_along_axis(func1d=em.compute_normalized_class_probas_doc,
                                  axis=em.vocab_axis,
                                  arr=unlabeled_count_data)

preds = np.argmax(pred_probas, axis=1)
correct_preds = preds == unlabeled_data_label_vals
pct_correct = np.sum(correct_preds) / len(unlabeled_data_label_vals)
print('model acc for unlabeled: %0.2f%%' % (100 * pct_correct))


