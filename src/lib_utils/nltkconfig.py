"""
Central location for the nltk_data.

Examples
#central location for nltk_data
kDefaultNLK_datafolder="d:/dev/ml/nce/ssl/datasets/nltk_data"


@author: mannykao
@editor: ivbarrie
"""

import os
from typing import List
from pathlib import Path, PurePosixPath
import nltk
from . import projconfig



def getDataFolder() -> Path:
	""" return '{kDevRoot}/datasets/nltk_data' """
	root = projconfig.getDataFolder()
	return root / 'nltk_data' 	#ssl/datasets/nltk_data

def set_default_datafolder(folder:str):
	kDefaultNLK_datafolder = folder

def check_nltk_data(datapkg:str) -> bool:
	""" check to see if 'datapkg' already exists within NLTK_data search path """
	result = False
	for folder in nltk.data.path:
		result = (folder == datapkg)
		if result:
			return True
	return result	


def set_nltk_datapath(mydatafolder: str, override=False) -> None:
	"""Add specified nltk data dir path to nltk."""
	if mydatafolder:
		# https://stackoverflow.com/questions/36382937/nltk-doesnt-add-nltk-data-to-search-path/36383314#36383314
		if override:
			nltk.data.path = [mydatafolder]
		else:	
			nltk.data.path.append(mydatafolder)
	nltk_datafolder = nltk.data.path
	# print(f"nltk_datafolder={nltk_datafolder}")
	# return nltk_datafolder
	return None


def stopwords_exists(parent_name: str = "corpora") -> bool:
	"""Check if stopwords in nltk data dir."""
	exists = False
	try:
		candidate_list = os.listdir(nltk.data.find(parent_name))
	except:
		return False
	if candidate_list is None or type(candidate_list) != list or len(candidate_list) == 0:
		return False
	else:
		exists = "stopwords" in os.listdir(nltk.data.find(parent_name))
	return exists


def download_stopwords(nltk_data_dir) -> None:
	"""Download stopwords to nltk (corpora) data dir."""
	print(f"Downloading stopwords")
	nltk.download("stopwords", download_dir=nltk_data_dir)
	return None


def get_english_stopwords(nltk_data_dir: str = "") -> List[str]:
	"""Retrieve english stopwords from nltk data dir."""
	if not stopwords_exists():
		download_stopwords(nltk_data_dir)
	en_stopwords = nltk.corpus.stopwords.words('english')
	return en_stopwords

if __name__ == '__main__':
	reporoot = projconfig.getRepoRoot()
	print(f"reporoot={reporoot}")
	datafolder = getDataFolder()
	print(f"datafolder={datafolder}")

	print(f"check_nltk_data('corpora')={check_nltk_data('corpora')}")
	print(f"check_nltk_data('corpora/words')={check_nltk_data('corpora/words')}")

