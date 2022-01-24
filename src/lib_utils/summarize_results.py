"""
Plot accuracy results from experiment

"""

from typing import Dict
from matplotlib import pyplot as plt


def plot_test_acc(test_acc_results: Dict[int, Dict[str, float]],
                  n_unlabeled: int,
                  plot_save_path: str = "test_acc_plot.png",
                  only_labeled_key: str = "only_labeled_train",
                  ssl_key: str = "ssl_train") -> None:
    """Save plot of labeled vs labeled + unlabeled model out-of-sample accuracy."""
    n_labeled_list = list(test_acc_results.keys())
    n_labeled_list.sort()
    only_labeled_acc_list = [100 * test_acc_results[k][only_labeled_key] for k in n_labeled_list]
    ssl_acc_list = [100 * test_acc_results[k][ssl_key] for k in n_labeled_list]
    plt.plot(n_labeled_list, only_labeled_acc_list, '*--', label="no unlabeled docs")
    plt.plot(n_labeled_list, ssl_acc_list, '*--', label="%d unlabeled docs" % n_unlabeled)
    plt.title("Out-of-sample test accuracy [7.5k]")
    plt.xlabel("Num labeled docs in training")
    plt.ylabel("Test accuracy [%]")
    plt.legend()
    print("saving plot to\n%s" % plot_save_path)
    plt.savefig(plot_save_path)

    return None
