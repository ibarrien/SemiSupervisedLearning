"""
Entry point.

@author: ivbarrie
"""


import pickle
import pathlib
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from .lib_utils.preprocessing import TextPreProcessor

# PARAMS
version = 1
english_vocab = set(nltk_english_words.words())
_tokens_to_remove = stopwords.words('english')
_tokens_to_remove.append('e')

# PATHS
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_output_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v%s.pkl' % version)
test_output_filepath = pathlib.PurePath(main_dir, 'test_preprocessed_v%s.pkl' % version)
processor = TextPreProcessor(tokens_to_remove=_tokens_to_remove,
                             english_vocab=english_vocab)

# EXECUTE
processor.set_train_raw_data()
processor.set_train_count_data()
processor.get_doc_lens()
scaled_train_data = processor.make_uniform_doc_lens(word_count_data=processor.train_count_data, strategy='median')
pickle.dump(scaled_train_data, open(train_output_filepath, 'wb'))
median, avg = processor.stats_nonzero_word_count_per_doc(word_count_data=processor.train_count_data)

