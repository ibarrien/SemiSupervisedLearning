# Semi Supervised Learning
Demonstrate how adding unlabeled data to a supervised problem can improve out-of-sample accuracy.

This repo provides an example of this accuracy improvement through 
a loss function := labeled_loss + unlabeled_loss which is optimized using expectation-maximization (EM).


Follows Nigam et al 2006 from "Semi-Supervised Learning", Chapelle et al.

## Running experiments
### Create virtual env
```python -m venv venv4ssl```

### Activate virtual env
Windows/Anaconda: venv4ssl/Script/activate

Powershell notes:

1. you may need to Powershell as administrator
2. \venv4ssl\Scripts\Activate.ps1

macos/Linux: source venv4ssl/bin/activate

Linux notes: installing matplotlib may require headers like ft2build.h
### Install requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### nltk_data note:
Handled by specifying nltk data dir in run_experiments_main

### run experiments with varying num of labeled samples:
```bash
python src/run_experiments_main.py 
--n_labeled 20,100,300,500,700,1000 
--n_unlabeled 10000 
--max_iters 5 
--out_dir <your_output_dir> 
--nltk_data_dir <optional: local nltk data dir> 
--test_acc_plot_fname test_acc.png
```

<img src="https://github.com/ibarrien/SemiSupervisedLearning/blob/main/src/acc_plot_uniform_train_sampling.png?raw=true" width="350" height="300" />
