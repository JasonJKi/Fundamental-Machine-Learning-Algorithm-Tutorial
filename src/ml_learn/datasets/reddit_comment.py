import numpy as np
import itertools
import csv
import nltk
import os, sys
from importlib import resources
from os.path import dirname, join as joinpath
from .. import DATA_DIR 
# DATA_DIR = joinpath(dirname(__file__), 'data')

DATA_MODULE = "ml_learn.datasets.data"
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


def _open_text(data_module, data_file_name):
    if sys.version_info >= (3, 9):
        return resources.files(data_module).joinpath(data_file_name).open("r")
    else:
        return resources.open_text(data_module, data_file_name)


def reddit_data(data_file_name='reddit-comments-2015-08.csv'):
# Read the data and append SENTENCE_START and SENTENCE_END tokens

    print("Reading CSV file...")
    data_path = DATA_DIR + '/' + data_file_name
    print(data_path)
    with open(data_path, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[0])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

    # Create the training data
    X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X, y