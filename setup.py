"""
Basic setup of packages.

$ python setup.py install

After setup install, text preprocessing, can do:

$ python src/em_driver.py

@author: ibarrien, mannykao
"""


import setuptools


setuptools.setup(name="Semi Supervised Learning, NLP prototype",
	version="1.02",
	description="Proof of concept SSL + EM",
	author="Ivan Barrientos",
	author_email="corps.des.nombres@gmail.com",
	packages=[
		"src",
		"lib_utils",
	],
	python_requires='>=3.6',
	install_requires=[
		"wheel",
		"nptyping==1.4.4",
		"numpy==1.20.2",
		"nltk",
		"sklearn",
	],
)
