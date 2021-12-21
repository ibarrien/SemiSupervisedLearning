"""
Main driver to run experiments.
Fixed: unlabeled dataset for EM; test dataset

Dynamic: labeled train dataset for EM
For each loop:
- increase labeled train dataset size
- evaluate on test set

@author: ibarrien
"""
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from lib_utils.preprocessing import TextPreProcessor
from lib_utils.expectation_maximization import EM_SSL


# Set tokens to remove for all text preprocessing
remove_zero_vocab_docs = True
english_vocab = set(nltk_english_words.words())
english_vocab = None
_tokens_to_remove = stopwords.words('english')
#_tokens_to_remove.append('e')

# Set labeled training data size range for experiments
n_labeled_train_samples_list = [int(200 * k) for k in range(1, 5)]

# Fix static preprocessed data
processor = TextPreProcessor(n_unlabeled_train_samples=1000,
                                    tokens_to_remove=_tokens_to_remove,
                                    remove_zero_vocab_docs=remove_zero_vocab_docs,
                                    english_vocab=english_vocab)
# Initialize raw
processor.set_static_full_train_raw_data()
processor.set_static_raw_unlabeled_data()
processor.set_static_raw_test_data()

# Execute experiments
for n_labeled_train_samples in n_labeled_train_samples_list:
    if n_labeled_train_samples == 0:
        print("Warning: experiment loop skipping zero labeled train samples")
        continue
    # set this train sample
    print("curr n labeled train samples: %d" % n_labeled_train_samples)
    processor.set_n_labeled_train_samples(n=n_labeled_train_samples)
    processor.set_sample_raw_train_data()

    # doc-to-vect based on train sample's count vectorizer
    processor.set_labeled_train_sample_count_data()
    print('labeled_train_sample_count_data shape:', processor.labeled_train_sample_count_data.shape)
    processor.set_unlabeled_count_data()
    print('unlabeld train count_data shape:', processor.unlabeled_count_data.shape)
    processor.set_test_count_data()

    # scale count data to trains' unif doc len
    scaled_labeled_train_sample_data = processor.make_uniform_doc_lens(word_count_data=processor.labeled_train_sample_count_data)
    scaled_unlabeled_data = processor.make_uniform_doc_lens(word_count_data=processor.unlabeled_count_data)
    scaled_test_data = processor.make_uniform_doc_lens(word_count_data=processor.test_count_data)

    # train
    model = EM_SSL(labeled_count_data=scaled_labeled_train_sample_data,
                   label_vals=processor.train_sample_label_vals,
                   unlabeled_count_data=scaled_unlabeled_data,
                   max_em_iters=10,
                   min_em_loss_delta=2e-4)

    model.fit()

    # out-of-sammple inference: test
    pct_test_correct_preds = model.evaluate_on_data(count_data=scaled_test_data,
                                                label_vals=processor.full_test_label_vals)
    print(pct_test_correct_preds)

