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
import argparse
import nltk

from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from lib_utils import nltkconfig
#
# https://www.nltk.org/data.html
#
nltk_data_list = [
	"corpora",
	"words",
]

if __name__ == '__main__':
	defaultNLTK_datafolder = nltkconfig.kDefaultNLK_datafolder
	#print(f"NLTK_data folder: {defaultNLTK_datafolder}")

	parser = argparse.ArgumentParser(description='NLTK data downloader')
	parser.add_argument('--data', type = str, default = defaultNLTK_datafolder, metavar=defaultNLTK_datafolder, help = "default folder for NLTK data")
	args = parser.parse_args()
	defaultNLTK_datafolder = args.data

	#1: see if the NLTK 'corpora' was downloaded
	downloaded = True
	for data in nltk_data_list:
		downloaded &= nltkconfig.check_nltk_data(data)

	if not downloaded:
		#2: invoke download dialog and setting the download folder to where we want it
		nltk.download(download_dir=defaultNLTK_datafolder)

	#3 overide nltk.data.path to our folder, you also can use append 
	nlk_datafolder = nltkconfig.NLTK_datapath(defaultNLTK_datafolder, override=True)
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
