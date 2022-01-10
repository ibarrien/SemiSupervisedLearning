"""
Central location for the nltk_data

Dynamic: labeled train dataset for EM
For each loop:
- increase labeled train dataset size
- evaluate on test set

@author: mannykao
"""
import os, sys
from pathlib import Path
import nltk


#central location for nltk_data
kDefaultNLK_datafolder="d:/Dev/datasets/nltk_data"	#TODO: temporaray hack to get everthing running. Do a proper projconfig.

def direxist(dirname):
	return os.path.isdir(dirname) and os.path.exists(dirname)

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
	print(f"{check_nltk_data('corpora')=}")
	print(f"{check_nltk_data('corpora/words')=}")

