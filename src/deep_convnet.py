'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.

'''

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Conv1D, GlobalMaxPooling1D
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

# Set parameters:
# ngram_range = 2 will add bi-grams features
TRAIN_MODE = True
NUM_EPOCHS = 5
NUM_FILTERS = 100
KERNEL_SIZE = 3
OUTPUT_SIZE = 128
MAX_FEATURES = 200000
EMBEDDING_SIZE = 300
DROPOUT_PROB = 0.5
MAX_LEN = 20
BATCH_SIZE = 128
PATIENCE = 5
WEIGHTS_NAME = 'weights_convnet.h5'
COMPILED_MODEL_NAME = 'compiled_convnet_model.h5'
TRAIN_OUTPUT_PROB_NAME = '../../data/train_convnet_proba.pkl'
GLOVE_VECTORDS_PATH = '../../data/glove.840B.300d.txt'
INPUT_TRAIN_FILE = '../../data/train.csv'
INPUT_TEST_FILE = '../../data/test.csv'
TEST_OUTPUT_PROB_NAME = '../../data/test_convnet_proba.pkl'

Q1_train = pd.read_csv('../../data/train_question1_stemmed.csv')
Q2_train = pd.read_csv('../../data/train_question2_stemmed.csv')

data = pd.DataFrame()
data['question1'] = Q1_train['0'].astype(str).values
data['question2'] = Q2_train['0'].astype(str).values
y = pd.read_csv('../../data/train.csv', usecols=['is_duplicate']).values
# data = pd.read_csv(INPUT_TRAIN_FILE, sep=',')
# y = data.is_duplicate.values

tk = text.Tokenizer(num_words=MAX_FEATURES)
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1_train = sequence.pad_sequences(tk.texts_to_sequences(data.question1.values), maxlen=MAX_LEN)
x2_train = sequence.pad_sequences(tk.texts_to_sequences(data.question2.values.astype(str)), maxlen=MAX_LEN)

word_index = tk.word_index
print('x_train shape:', x1_train.shape)
print('x_test shape:', x2_train.shape)


def make_convnet_model(embedding_matrix=None, weights_path=None):
    model1 = Sequential()
    model1.add(Embedding(len(word_index) + 1,
                         EMBEDDING_SIZE,
                         weights=[embedding_matrix], trainable=False,
                         input_length=MAX_LEN))
    model1.add(Dropout(DROPOUT_PROB))
    model1.add(Conv1D(NUM_FILTERS,
                      KERNEL_SIZE,
                      padding='valid',
                      activation='relu',
                      strides=1))
    model1.add(GlobalMaxPooling1D())
    model1.summary()

    model2 = Sequential()
    model2.add(Embedding(len(word_index) + 1,
                         EMBEDDING_SIZE,
                         weights=[embedding_matrix], trainable=False,
                         input_length=MAX_LEN))
    model2.add(Dropout(DROPOUT_PROB))
    model2.add(Conv1D(NUM_FILTERS,
                      KERNEL_SIZE,
                      padding='valid',
                      activation='relu',
                      strides=1))
    model2.add(GlobalMaxPooling1D())
    model2.summary()
    merged_model = Sequential()
    merged_model.add(Merge([model1, model2], mode='concat'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(OUTPUT_SIZE))
    merged_model.add(PReLU())
    merged_model.add(Dropout(DROPOUT_PROB))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(1, activation='sigmoid'))
    if weights_path is not None:
        print 'Restoring weights ... '
        merged_model.load_weights(weights_path)
    merged_model.compile(loss='binary_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    merged_model.summary()
    return merged_model


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
    merged_model = make_convnet_model(embedding_matrix, WEIGHTS_NAME)
    checkpoint = ModelCheckpoint(WEIGHTS_NAME, monitor='val_loss', save_best_only=True, verbose=2)
    earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0)
    merged_model.fit(input_arrays, y, batch_size=BATCH_SIZE,
                     epochs=NUM_EPOCHS, validation_split=0.25, shuffle=True, callbacks=[earlystop, checkpoint])

    train_proba = merged_model.predict(input_arrays, batch_size=BATCH_SIZE)
    cPickle.dump(train_proba.ravel(), open(TRAIN_OUTPUT_PROB_NAME, 'wb'), -1)
    merged_model.save(COMPILED_MODEL_NAME)
else:
    trained_model = make_convnet_model(embedding_matrix, WEIGHTS_NAME)
    data_test = pd.read_csv(INPUT_TEST_FILE, sep=',')
    x1_test = tk.texts_to_sequences(data_test.question1.values.astype(str))
    x2_test = tk.texts_to_sequences(data_test.question2.values.astype(str))
    x1_test = sequence.pad_sequences(x1_test, maxlen=MAX_LEN)
    x2_test = sequence.pad_sequences(x2_test, maxlen=MAX_LEN)

    trained_model.load_weights(WEIGHTS_NAME)
    test_proba = trained_model.predict([x1_test, x2_test], batch_size=BATCH_SIZE)
    test_proba = test_proba.ravel()
    cPickle.dump(test_proba, open(TEST_OUTPUT_PROB_NAME, 'wb'), -1)

    make_submission(test_proba)
