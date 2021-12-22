"""
Debug text preprocessing (prior to model training)

@author ivbarrie
"""

from nltk.corpus import stopwords
from nltk.corpus import words as nltk_english_words

from lib_utils.preprocessing import TextPreProcessor


# Set tokens to remove for all text preprocessing
remove_zero_vocab_docs = True
english_vocab = set(nltk_english_words.words())
english_vocab = None
_tokens_to_remove = stopwords.words('english')


# Fix static preprocessed data
# original article suggests 10k fixed unlabeled samples
processor = TextPreProcessor(n_unlabeled_train_samples=10000,
                                    tokens_to_remove=_tokens_to_remove,
                                    remove_zero_vocab_docs=remove_zero_vocab_docs,
                                    english_vocab=english_vocab)
# Initialize raw
processor.set_static_full_train_raw_data()
processor.set_static_raw_unlabeled_data()
processor.set_static_raw_test_data()

# Sample text preprocessing
# Examples:
# Input: Do YOU eat all your food cold?
# Output: ['eat food cold']
processor.set_sample_raw_train_data()
sample_sent = processor.labeled_train_data_sample[0]
print('sample labeled trained sentence\n', sample_sent)
processed_sent = processor.process_documents_text(sample_sent)
print('processed sentence\n', processed_sent)

doc_array_sample = processor.labeled_train_data_sample[:2]
processed_docs = processor.process_documents_text(doc_array_sample)  # List[str]


