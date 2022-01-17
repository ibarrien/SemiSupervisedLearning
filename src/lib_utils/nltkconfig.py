"""
Central location for the nltk_data

@author: mannykao
"""
import os, sys
#import re
from pathlib import Path, PurePosixPath

import nltk

from . import projconfig

#central location for nltk_data
kDefaultNLK_datafolder="d:/dev/ml/nce/ssl/datasets/nltk_data"


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
		return True
	return result	

def NLTK_datapath(mydatafolder:str=kDefaultNLK_datafolder, override=False):
	if mydatafolder:
		# https://stackoverflow.com/questions/36382937/nltk-doesnt-add-nltk-data-to-search-path/36383314#36383314
		if override:
			nltk.data.path = [mydatafolder]
		else:	
			nltk.data.path.append(mydatafolder)
	nlk_datafolder = nltk.data.path
	print(f"{nlk_datafolder=}")
	return nlk_datafolder


if __name__ == '__main__':
	reporoot = projconfig.getRepoRoot()
	print(f"{reporoot=}")
	datafolder = getDataFolder()
	print(f"{datafolder=}")

	print(f"{check_nltk_data('corpora')=}")
	print(f"{check_nltk_data('corpora/words')=}")
