import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers.core import Lambda, Flatten
from keras.engine import Input
from keras.engine import Layer
from keras.engine import Model

from keras.layers import Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, Lambda, dot, merge
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
from tqdm import tqdm
from keras import backend as K

# Set parameters:
# ngram_range = 2 will add bi-grams features
TRAIN_MODE = True
NUM_EPOCHS = 15
OUTPUT_SIZE = 128
MAX_FEATURES = 200000
EMBEDDING_SIZE = 300
DROPOUT_PROB = 0.5
MAX_LEN = 40
BATCH_SIZE = 128
PATIENCE = 5
WEIGHTS_NAME = 'weights_dualencoder.h5'
COMPILED_MODEL_NAME = 'compiled_dualencoder_model.h5'
TRAIN_OUTPUT_PROB_NAME = '../../data/train_dualencoder_proba.pkl'
GLOVE_VECTORDS_PATH = '../../data/glove.840B.300d.txt'
INPUT_TRAIN_FILE = '../../data/train.csv'
INPUT_TEST_FILE = '../../data/test.csv'
TEST_OUTPUT_PROB_NAME = '../../data/test_dualencoder_proba.pkl'


data = pd.read_csv(INPUT_TRAIN_FILE, sep=',')
y = data.is_duplicate.values
y = y.reshape(y.shape[0], 1)

tk = text.Tokenizer(num_words=MAX_FEATURES)
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1_train = sequence.pad_sequences(tk.texts_to_sequences(data.question1.values), maxlen=MAX_LEN)
x2_train = sequence.pad_sequences(tk.texts_to_sequences(data.question2.values.astype(str)), maxlen=MAX_LEN)

word_index = tk.word_index
print('x_train shape:', x1_train.shape)
print('x_test shape:', x2_train.shape)


def make_dualencoders(embedding_matrix=None, weights_path=None):
    def create_base_network(flag):
        '''Base network to be shared (eq. to feature extraction).
        '''
        model1 = Sequential()
        model1.add(Embedding(len(word_index) + 1,
                             EMBEDDING_SIZE,
                             weights=[embedding_matrix], trainable=False,
                             input_length=MAX_LEN))
        model1.add(Dropout(DROPOUT_PROB))
        model1.add(Conv1D(100,
                          5,
                          padding='valid',
                          activation='relu',
                          strides=1))
        model1.add(GlobalMaxPooling1D())
        return model1

    base_network = create_base_network(MAX_LEN)

    input_a = Input(shape=(MAX_LEN,))
    input_b = Input(shape=(MAX_LEN,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    dot = merge([processed_a, processed_b], mode='dot', dot_axes=1)
    model = Model(input=[input_a, input_b], output=dot)

    if weights_path is not None:
        print 'Restoring weights ... '
        model.load_weights(weights_path)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def make_submission(test_proba):
    submission = pd.read_csv('../../data/sample_submission.csv')
    defaults = submission.is_duplicate.values.astype(float)
    defaults = defaults.ravel()

    for idx, (old, new) in enumerate(itertools.izip_longest(defaults, test_proba, fillvalue='0.5')):
        if new == '0.5':
            new = random.uniform(0.000001, 0.9999999)
        defaults[idx] = float(new)

    submission['is_duplicate'] = pd.Series(defaults, index=submission.index).astype(float)
    import time

    submission.to_csv('../../submissions/submission_{0}.csv'.format(time.time()), index=False)


embeddings_index = {}
f = open(GLOVE_VECTORDS_PATH)
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_SIZE))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

if TRAIN_MODE:
    input_arrays = [x1_train, x2_train]
    merged_model = make_dualencoders(embedding_matrix, None)
    checkpoint = ModelCheckpoint(WEIGHTS_NAME, monitor='val_loss', save_best_only=True, verbose=2)
    earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0)
    merged_model.fit(input_arrays, y,
                     batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCHS,
                     validation_split=0.1,
                     shuffle=True, callbacks=[earlystop, checkpoint])

    # train_proba = merged_model.predict(input_arrays, batch_size=BATCH_SIZE)
    # cPickle.dump(train_proba.ravel(), open(TRAIN_OUTPUT_PROB_NAME, 'wb'), -1)
    merged_model.save(COMPILED_MODEL_NAME)
# else:
#     trained_model = make_dualencoders(embedding_matrix, WEIGHTS_NAME)
#     data_test = pd.read_csv(INPUT_TEST_FILE, sep=',')
#     x1_test = tk.texts_to_sequences(data_test.question1.values.astype(str))
#     x2_test = tk.texts_to_sequences(data_test.question2.values.astype(str))
#     x1_test = sequence.pad_sequences(x1_test, maxlen=MAX_LEN)
#     x2_test = sequence.pad_sequences(x2_test, maxlen=MAX_LEN)
#
#     trained_model.load_weights(WEIGHTS_NAME)
#     test_proba = trained_model.predict([x1_test, x2_test], batch_size=BATCH_SIZE)
#     test_proba = test_proba.ravel()
#     cPickle.dump(test_proba, open(TEST_OUTPUT_PROB_NAME, 'wb'), -1)
#
#     make_submission(test_proba)
