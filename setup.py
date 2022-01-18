"""
Basic setup of packages.

$ python setup.py install

@author: ibarrien
"""


from distutils.core import setup


setup(name="Semi Supervised Learning, NLP prototype",
	version="1.02",
	description="Proof of concept SSL + EM",
	author="Ivan Barrientos",
	author_email="corps.des.nombres@gmail.com",
	packages=[
		"lib_utils",
	],
	install_requires=[
		"nptyping==1.4.4",
		"numpy>=1.21",
	],
)
