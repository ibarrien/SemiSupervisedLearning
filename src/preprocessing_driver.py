"""
Text preprocessing entry point.

@author: ivbarrie
"""


import pickle
import pathlib
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

#from .lib_utils.preprocessing import TextPreProcessor

# PARAMS
version = 'all_train_english_filter'
train_pct_subsample = 1.0
remove_zero_vocab_docs = True
english_vocab = set(nltk_english_words.words())
# english_vocab = None
_tokens_to_remove = stopwords.words('english')
_tokens_to_remove.append('e')

# PATHS
main_dir = r'C:/Users/ivbarrie/Desktop/Projects/SSL'
train_data_output_filepath = pathlib.PurePath(main_dir, 'train_preprocessed_v_%s.pkl' % version)
train_labels_output_filepath = pathlib.PurePath(main_dir, 'train_labels_v_%s.pkl' % version)
train_count_vectorizer_filepath = pathlib.PurePath(main_dir, 'trained_count_vectorizer_v%s.pkl' % version)
test_output_filepath = pathlib.PurePath(main_dir, 'test_preprocessed_v%s.pkl' % version)
processor = TextPreProcessor(tokens_to_remove=_tokens_to_remove,
                             train_pct_subsample=train_pct_subsample,
                             remove_zero_vocab_docs=remove_zero_vocab_docs,
                             english_vocab=english_vocab)

# EXECUTE
processor.set_train_raw_data()
processor.set_train_count_data()  # train_count_data, train_label_vals, train_label_names
#
processor.get_doc_lens()
scaled_train_data = processor.make_uniform_doc_lens(word_count_data=processor.train_count_data, strategy='median')
pickle.dump(scaled_train_data, open(train_data_output_filepath, 'wb'))
pickle.dump(processor.train_label_vals, open(train_labels_output_filepath, 'wb'))
median, avg = processor.stats_nonzero_word_count_per_doc(word_count_data=processor.train_count_data)

