"""
Basic setup of packages.

$ python setup.py install

After setup install, text preprocessing, can do:

$ python src/em_driver.py

@author: ibarrien, mannykao
"""
import setuptools

#
# this is not yet needed since we are not a package. See README.md.
#

setuptools.setup(name="Semi Supervised Learning, NLP prototype",
	version="1.02",
	description="Proof of concept SSL + EM",
	author="Ivan Barrientos",
	author_email="corps.des.nombres@gmail.com",
	packages=[
		"src",
		#"lib_utils",
	],
	python_requires='>=3.8',
	install_requires=[
		"wheel",
		"nptyping==1.4.4",
		"numpy==1.21.5",
		"nltk==3.6.6",
		"scikit-learn==0.23.2",
	],
)
