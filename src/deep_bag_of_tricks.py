from __future__ import print_function
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Merge, BatchNormalization, PReLU
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
import pandas as pd
from keras.preprocessing import sequence, text
import cPickle
import itertools
import random
from keras.models import load_model


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
TRAIN_MODE = False
NGRAM_RANGE = 2
NUM_EPOCHS = 15
OUTPUT_SIZE = 128
MAX_FEATURES = 200000
EMBEDDING_SIZE = 300
DROPOUT_PROB = 0.5
MAX_LEN = 40
BATCH_SIZE = 64
PATIENCE = 5
WEIGHTS_NAME = 'weights_bagoftricks.h5'
COMPILED_MODEL_NAME = 'compiled_bagoftricks_model.h5'
TRAIN_OUTPUT_PROB_NAME = '../../data/train_bagoftricks_proba.pkl'
TEST_OUTPUT_PROB_NAME = '../../data/test_bagoftricks_proba.pkl'
GLOVE_VECTORDS_PATH = '../../data/glove.840B.300d.txt'
INPUT_TRAIN_FILE = '../../data/train.csv'
INPUT_TEST_FILE = '../../data/test.csv'

data = pd.read_csv(INPUT_TRAIN_FILE, sep=',')
y = data.is_duplicate.values

tk = text.Tokenizer(num_words=MAX_FEATURES)
tk.fit_on_texts(list(data.question1.values.astype(str)) + list(data.question2.values.astype(str)))
x1_train = tk.texts_to_sequences(data.question1.values.astype(str))
x2_train = tk.texts_to_sequences(data.question2.values.astype(str))

word_index = tk.word_index

token_indice = None
indice_token = None
ngram_set = None

if NGRAM_RANGE > 1:
    print('Adding {}-gram features'.format(NGRAM_RANGE))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in itertools.chain(x1_train, x2_train):
        for i in range(2, NGRAM_RANGE + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = MAX_FEATURES + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x1_train = add_ngram(x1_train, token_indice, NGRAM_RANGE)
    x2_train = add_ngram(x2_train, token_indice, NGRAM_RANGE)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, x1_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, x2_train)), dtype=int)))

print('Pad sequences (samples x time)')
x1_train = sequence.pad_sequences(x1_train, maxlen=MAX_LEN)
x2_train = sequence.pad_sequences(x2_train, maxlen=MAX_LEN)
print('x_train shape:', x1_train.shape)
print('x_test shape:', x2_train.shape)


def make_bagoftricks_model(weights_path=None):
    model1 = Sequential()
    model1.add(Embedding(len(word_index) + 1, EMBEDDING_SIZE, input_length=MAX_LEN))
    model1.add(GlobalAveragePooling1D())

    model2 = Sequential()
    model2.add(Embedding(len(word_index) + 1, EMBEDDING_SIZE, input_length=MAX_LEN))
    model2.add(GlobalAveragePooling1D())

    merged_model = Sequential()
    merged_model.add(Merge([model1, model2], mode='concat'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(OUTPUT_SIZE))
    merged_model.add(PReLU())
    merged_model.add(Dropout(DROPOUT_PROB))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(1, activation='sigmoid'))
    if weights_path is not None:
        merged_model.load_weights(weights_path)
    merged_model.compile(loss='binary_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    merged_model.summary()
    return merged_model


def make_submission(test_proba):
    submission = pd.read_csv('../../data/sample_submission.csv')
    defaults = submission.is_duplicate.values.astype(float)
    defaults = defaults.ravel()

    for idx, (old, new) in enumerate(itertools.izip_longest(defaults, test_proba, fillvalue='0.165')):
        if new == '0.165':
            new = random.uniform(0.000001, 0.9999999)
        defaults[idx] = float(new)

    submission['is_duplicate'] = pd.Series(defaults, index=submission.index).astype(float)
    import time

    submission.to_csv('../../submissions/submission_{0}.csv'.format(time.time()), index=False)


# Train model
input_arrays = [x1_train, x2_train]
merged_model = make_bagoftricks_model()
checkpoint = ModelCheckpoint(WEIGHTS_NAME, monitor='val_loss', save_best_only=True, verbose=2)
earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0)
merged_model.fit(input_arrays, y, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, validation_split=0.01, shuffle=True, callbacks=[earlystop, checkpoint])

# Predict on train dataset
train_proba = merged_model.predict(input_arrays, batch_size=BATCH_SIZE)
cPickle.dump(train_proba.ravel(), open(TRAIN_OUTPUT_PROB_NAME, 'wb'), -1)
merged_model.save(COMPILED_MODEL_NAME)

# Load test dataset
data_test = pd.read_csv(INPUT_TEST_FILE, sep=',')
x1_test = tk.texts_to_sequences(data_test.question1.values.astype(str))
x1_test = add_ngram(x1_test, token_indice, NGRAM_RANGE)
x1_test = sequence.pad_sequences(x1_test, maxlen=MAX_LEN)
x2_test = tk.texts_to_sequences(data_test.question2.values.astype(str))
x2_test = add_ngram(x2_test, token_indice, NGRAM_RANGE)
x2_test = sequence.pad_sequences(x2_test, maxlen=MAX_LEN)

# Predict on test
# trained_model.load_weights(WEIGHTS_NAME)
test_proba = merged_model.predict([x1_test, x2_test], batch_size=BATCH_SIZE)
test_proba = test_proba.ravel()
cPickle.dump(test_proba, open(TEST_OUTPUT_PROB_NAME, 'wb'), -1)

# Make submission
make_submission(test_proba)
