# Semi Supervised Learning

### Example of applying expectation maximization to loss function := labeled_loss + unlabeled_loss

### Follows Nigam et al 2006 from "Semi-Supervised Learning", Chapelle et al.

## Running an experiment:

# python -m venv venv4ssl
# Windows/Anaconda: venv4ssl/Script/activate
# macos/Linux: source venv4ssl/bin/activate
# python -m pip install --upgrade pip

# install our requirements:
$ pip install -r requirements.txt

## nltk_data note:
Handled by specifying nltk data dir in run_experiments_main

# run experiments with varying num of labeled samples:
```
python src/run_experiments_main.py 
--n_labeled 100,300,500,700,1000 
--n_unlabeled 10000 
--max_iters 10 
--out_dir <your_output_dir> 
--nltk_data_dir <optional: local nltk data dir> 
--test_acc_plot_fname test_acc.png
```

<img src="https://user-images.githubusercontent.com/7552335/150062562-64e3c1fe-3f08-4dac-80c5-b9eb13e5c4fc.png" width="350" height="300" />
