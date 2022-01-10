"""
Main driver to run experiments.
Fixed: unlabeled dataset for EM; test dataset

Dynamic: labeled train dataset for EM
For each loop:
- increase labeled train dataset size
- evaluate on test set

@author: mannykao
"""
#from pathlib import Path
import nltk

from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from lib_utils import nltkconfig
#
# https://www.nltk.org/data.html
#

if __name__ == '__main__':
	defaultNLK_datafolder = nltkconfig.kDefaultNLK_datafolder	#TODO: control from command line using argparse

	#1: see if the NLYK 'corpora' was downloaded
	downloaded = nltkconfig.check_nltk_data("corpora")

	if not downloaded:
		#2: invoke download dialog and setting the download folder to where we want it
		nltk.download(download_dir=defaultNLK_datafolder)

	#3 overide nltk.data.path to our folder, you also can use append 
	nlk_datafolder = nltkconfig.NLTK_datapath(defaultNLK_datafolder, override=True)
	try:
		downloaded = nltk.download("words")
	except:
		print(f"Failed to download {nltk.corpus.words}, run nltk.download().")
		# NLTK Data download dialog

	# Set tokens to remove for all text preprocessing
	remove_zero_vocab_docs = True
	english_vocab = set(nltk_english_words.words())
	print(f"{len(english_vocab)=}")
	english_vocab = None
	_tokens_to_remove = stopwords.words('english')
	print(f"{len(_tokens_to_remove)=}")
