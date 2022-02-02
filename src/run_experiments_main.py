"""
Main driver to run experiments.
Fixed: unlabeled dataset for EM; test dataset

Dynamic: labeled train dataset for EM
For each loop:
- increase labeled train dataset size
- evaluate on fixed test set

@authors: ibarrien
"""
import argparse
import pathlib
from lib_utils import nltkconfig, torchutils
from lib_utils.preprocessing import TextPreProcessor
from lib_utils.expectation_maximization import EM_SSL
from lib_utils.summarize_results import plot_test_acc


def main(args):
    nltk_data_dir = args.nltk_data_dir
    nltkconfig.set_nltk_datapath(mydatafolder=nltk_data_dir,
                                 override=True)
    english_stopwords = nltkconfig.get_english_stopwords(nltk_data_dir=nltk_data_dir)
    remove_zero_vocab_docs = False
    english_vocab = None

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
    rmv_fields = []
    processor = TextPreProcessor(n_unlabeled_train_samples=n_unlabeled_train,
                                 tokens_to_remove=english_stopwords,
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
        print(f" using {n_labeled_train_samples} labeled train samples: ")
        processor.set_n_labeled_train_samples(n=n_labeled_train_samples)
        processor.set_sample_raw_train_data()

        # doc-to-vect based on train sample's count vectorizer
        processor.set_labeled_train_sample_count_data()
        print(' labeled_train_sample_count_data shape:', processor.labeled_train_sample_count_data.shape)
        unique_sampled_train_label_vals = len(set(processor.train_sample_label_vals))
        if unique_sampled_train_label_vals < 20:
            print(f"sampled only {unique_sampled_train_label_vals} labels")
            print(f"Sampled less than 20 unique labels, skipping")
            continue
        processor.set_unlabeled_count_data()
        print(' unlabeled train count_data shape:', processor.unlabeled_count_data.shape)
        processor.set_test_count_data()

        # scale count data to train data unif doc len
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
        print(f" test_acc_results: {test_acc_results}")
        print()

    # PLOT SUMMARY RESULTS
    plot_test_acc(test_acc_results=test_acc_results,
                  n_unlabeled=n_unlabeled_train,
                  plot_save_path=test_acc_plot_path,
                  only_labeled_key="only_labeled_train",
                  ssl_key="ssl_train")


if __name__ == '__main__':
    torchutils.initSeeds()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_labeled', type=str, dest='n_labeled',
                        help='Comma separated no-spaces list of num labeled samples for training; '
                        'disjoint from unlabeled train samples.\n'
                        'Example: 500,1000,1500,2000\n'
                        'Note: each value determines an experiment'
                        'Note: default label sampling is uniform; alternative is emperical\n',
                        default='20,100,300,500,700,1000')
    parser.add_argument('--n_unlabeled', type=int, dest='n_unlabeled',
                        help='Num unlabeled samples for training; '
                        'disjoint from labeled train samples.\n'
                        'Note: each experiment uses the same number of unlabeled samples,'
                        'but not necc the same subset since disjointness enforced in sampling',
                        default=10000)
    parser.add_argument('--max_iters', type=int, dest='max_iters',
                        help='max number of EM iterations per experiment;'
                        'same as num epochs since each iter uses all training data',
                        default=5)
    parser.add_argument('--min_delta', type=float, dest='min_delta',
                        help='min improvement in loss between EM steps; '
                        'triggers early stopping if min delta improvement not met',
                        default=1e-2)
    parser.add_argument('--out_dir', type=str, dest='out_dir',
                        help='output folder name for experiment result plots',
                        default='')
    parser.add_argument('--nltk_data_dir', type=str, dest='nltk_data_dir',
                        help='path to nltk data, including corpora/stopwords.\
                        If corpora/stopwords does not exist, download here.',
                        default=nltkconfig.getDataFolder())
    parser.add_argument('--test_acc_plot_fname', type=str, dest='test_acc_plot_fname',
                        help='filename of test accuracy plot',
                        default='acc_plot_uniform_train_sampling.png')

    args = parser.parse_args()
    main(args)

