# Semi Supervised Learning

### Example of applying expectation maximization to loss function := labeled_loss + unlabeled_loss

### Follows Nigam et al 2006 from "Semi-Supervised Learning", Chapelle et al.

## Running experiments:
python setup.py install

### Run experiments when using number of labeled samples: 100,300,500,700,1000
python src/run_experiments_main.py --n_
labeled 100,300,500,700,1000 --n_unlabeled 10000 --max_iters 10 --out_dir <your_dir>  --test_acc_plot
_fname test_acc_rmv_zero_docs_keep_headers_footers_quotes.png

