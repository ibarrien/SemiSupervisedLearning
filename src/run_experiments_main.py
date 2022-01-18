"""
Main driver to run experiments.
Fixed: unlabeled dataset for EM; test dataset

Dynamic: labeled train dataset for EM
For each loop:
- increase labeled train dataset size
- evaluate on test set

@author: ibarrien
"""
import argparse
import pathlib
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from lib_utils.preprocessing import TextPreProcessor
from lib_utils.expectation_maximization import EM_SSL
from lib_utils.summarize_results import plot_test_acc

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_labeled', type=str, dest='n_labeled',
                        help='Comma separated no-spaces list of num labeled samples for training; '
                        'disjoint from unlabeled train samples.\n'
                        'Example: 500,1000,1500,2000\n'
                        'Note: each value determines an experiment'
                        'Note: uniform labeled distribution is sampled\n',
                        default='100,200')
    parser.add_argument('--n_unlabeled', type=int, dest='n_unlabeled',
                        help='Num unlabeled samples for training; '
                        'disjoint from labeled train samples.\n'
                        'Note: each experiment uses the same number of unlabeled samples,'
                        'but not necc the same subset since disjointness enforced in sampling',
                        default=5000)
    parser.add_argument('--max_iters', type=int, dest='max_iters',
                        help='max number of EM iterations per experiment;'
                        'same as num epochs since each iter uses all training data',
                        default=10)
    parser.add_argument('--min_delta', type=float, dest='min_delta',
                        help='min improvement in loss between EM steps; '
                        'triggers early stopping if min delta improvement not met',
                        default=1e-2)
    parser.add_argument('--out_dir', type=str, dest='out_dir',
                        help='output folder name for experiment result plots',
                        default='')

    parser.add_argument('--test_acc_plot_fname', type=str, dest='test_acc_plot_fname',
                        help='filename of test accuracy plot',
                        default='acc_plot.png')

    # TODO: add to parser
    # Set tokens to remove for all text preprocessing
    remove_zero_vocab_docs = True
    # english_vocab = set(nltk_english_words.words())
    english_vocab = None
    _tokens_to_remove = stopwords.words('english')

    args = parser.parse_args()

    # Set labeled training data size range for experiments
    n_labeled_train_samples_list = [int(s.strip()) for s in args.n_labeled.split(",")]  # [200, 500]
    print('n labeled list, \n', n_labeled_train_samples_list)
    # Set unlabeled (static) data size
    n_unlabeled_train = args.n_unlabeled
    # Set EM params
    max_iters = args.max_iters
    min_delta = args.min_delta
    # Output params
    test_acc_plot_path = pathlib.PurePath(args.out_dir, args.test_acc_plot_fname)


    # Fix static preprocessed data
    # original article suggests 10k fixed unlabeled samples
    rmv_fields = ('headers','footers','quotes')
    rmv_fields = []
    processor = TextPreProcessor(n_unlabeled_train_samples=n_unlabeled_train,
                                 tokens_to_remove=_tokens_to_remove,
                                 remove_zero_vocab_docs=remove_zero_vocab_docs,
                                 english_vocab=english_vocab,
                                 remove_fields=rmv_fields)
    # Initialize raw
    processor.set_static_full_train_raw_data()
    processor.set_static_raw_unlabeled_data()
    processor.set_static_raw_test_data()

    # Execute experiments
    test_acc_results = {}  # keys: n_labels, values: {only_labeled_acc, ssl_acc}
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
                       test_count_data=scaled_test_data,
                       test_label_vals=processor.full_test_label_vals,
                       max_em_iters=max_iters,
                       min_em_loss_delta=min_delta)

        model.fit()
        test_acc_results[n_labeled_train_samples] = {"only_labeled_train": model.only_labeled_test_acc(),
                                                     "ssl_train": model.last_em_iter_test_acc()}
        print(test_acc_results)

    # SUMMARIZE RESULTS
    acc_plot_name = "ssl_test_acc.png"
    plot_save_path = str(test_acc_plot_path)
    plot_test_acc(test_acc_results=test_acc_results,
                  n_unlabeled=n_unlabeled_train,
                  plot_save_path=test_acc_plot_path,
                  only_labeled_key="only_labeled_train",
                  ssl_key="ssl_train")


