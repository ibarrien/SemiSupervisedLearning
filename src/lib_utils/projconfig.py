"""
Central location for the training and test data sets

@author: mannykao
"""
import os, sys
import re
from pathlib import Path, PurePosixPath


#central location for nltk_data
kDevRoot = "ssl" 	#the same of our top level repo
#kDefaultNLK_datafolder = "d:/dev/ml/nce/ssl/datasets/nltk_data"

def direxist(dirname):
	return os.path.isdir(dirname) and os.path.exists(dirname)

def getRepoRoot() -> Path:
	""" return {kDevRoot} where {kDevRoot} is located - e.g. '<srcroot>/ssl' 
	"""
	root = ''
	ourpath = Path(__file__) 	#D:\Dev\ML\NCE\SSL\venv4ssl\lib\site-packages\shnetutil\projconfig.py
	posix = PurePosixPath(ourpath)

	for parent in posix.parents:
		if parent.name.lower() == kDevRoot:
			root = parent
	return root

def getDataFolder() -> Path:
	""" return '{kDevRoot}/datasets' """
	root = getRepoRoot()
	return root / 'datasets'
