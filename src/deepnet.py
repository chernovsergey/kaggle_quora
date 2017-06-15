import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import pandas as pd
import numpy as np
from keras.legacy.layers import Merge
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
import cPickle
import itertools
import random
from keras.models import load_model

TRAIN_MODE = True
MAX_FEATURES = 200000
FILTER_LEN = 5
NUM_FILTERS = 64
POOL_LEN = 4
EMBEDDING_SIZE = 300
OUTPUT_SIZE = 128
DROPOUT_PROB = 0.5
MAX_LEN = 20
PATIENCE = 5
BATCH_SIZE = 128
NUM_EPOCHS = 25
WEIGHTS_NAME = 'weights_monster.h5'
COMPILED_MODEL_NAME = 'compiled_monster_model.h5'
TRAIN_OUTPUT_PROB_NAME = '../../data/train_proba.pkl'
GLOVE_VECTORDS_PATH = '../../data/glove.840B.300d.txt'
INPUT_TRAIN_FILE = '../../data/train_small.csv'
INPUT_TEST_FILE = '../../data/test_small.csv'
TEST_OUTPUT_PROB_NAME = '../../data/test_proba.pkl'


def make_monster_model(embedding_matrix, word_index, train_embeddings=False, weights_path=None):
    # # # ================== 1 =======================
    # model1 = Sequential()
    # model1.add(Embedding(len(word_index) + 1,
    #                      EMBEDDING_SIZE,
    #                      weights=[embedding_matrix],
    #                      input_length=MAX_LEN,
    #                      trainable=train_embeddings))
    # model1.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='relu')))
    # model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(OUTPUT_SIZE,)))
    #
    # # ================== 2 =======================
    # model2 = Sequential()
    # model2.add(Embedding(len(word_index) + 1,
    #                      EMBEDDING_SIZE,
    #                      weights=[embedding_matrix],
    #                      input_length=MAX_LEN,
    #                      trainable=train_embeddings))
    # model2.add(TimeDistributed(Dense(OUTPUT_SIZE, activation='relu')))
    # model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(OUTPUT_SIZE,)))

    # ================== 3 =======================
    model3 = Sequential()
    model3.add(Embedding(len(word_index) + 1,
                         EMBEDDING_SIZE,
                         weights=[embedding_matrix],
                         input_length=MAX_LEN,
                         trainable=train_embeddings))
    model3.add(Convolution1D(nb_filter=NUM_FILTERS,
                             filter_length=FILTER_LEN,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model3.add(Dropout(DROPOUT_PROB))
    model3.add(Convolution1D(nb_filter=NUM_FILTERS,
                             filter_length=FILTER_LEN,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model3.add(GlobalMaxPooling1D())
    model3.add(Dropout(DROPOUT_PROB))
    model3.add(Dense(OUTPUT_SIZE))
    model3.add(Dropout(DROPOUT_PROB))
    model3.add(BatchNormalization())

    # ================== 4 =======================
    model4 = Sequential()
    model4.add(Embedding(len(word_index) + 1,
                         EMBEDDING_SIZE,
                         weights=[embedding_matrix],
                         input_length=MAX_LEN,
                         trainable=train_embeddings))
    model4.add(Convolution1D(nb_filter=NUM_FILTERS,
                             filter_length=FILTER_LEN,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model4.add(Dropout(DROPOUT_PROB))
    model4.add(Convolution1D(nb_filter=NUM_FILTERS,
                             filter_length=FILTER_LEN,
                             border_mode='valid',
                             activation='relu',
                             subsample_length=1))
    model4.add(GlobalMaxPooling1D())
    model4.add(Dropout(DROPOUT_PROB))
    model4.add(Dense(OUTPUT_SIZE))
    model4.add(Dropout(DROPOUT_PROB))
    model4.add(BatchNormalization())

    # ================== 5 =======================
    model5 = Sequential()
    model5.add(Embedding(len(word_index) + 1, EMBEDDING_SIZE, input_length=MAX_LEN, dropout=DROPOUT_PROB,
                         trainable=train_embeddings))
    model5.add(LSTM(EMBEDDING_SIZE, dropout_W=DROPOUT_PROB, dropout_U=DROPOUT_PROB))

    # ================== 6 =======================
    model6 = Sequential()
    model6.add(Embedding(len(word_index) + 1, EMBEDDING_SIZE, input_length=MAX_LEN, dropout=DROPOUT_PROB,
                         trainable=train_embeddings))
    model6.add(LSTM(EMBEDDING_SIZE, dropout_W=DROPOUT_PROB, dropout_U=DROPOUT_PROB))

    # ================== Merge all =======================
    models_to_merge = [model3, model4, model5, model6]
    merged_model = Sequential()
    merged_model.add(Merge(models_to_merge, mode='concat'))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(OUTPUT_SIZE))
    merged_model.add(PReLU())
    # merged_model.add(Dropout(DROPOUT_PROB))
    # merged_model.add(BatchNormalization())
    # merged_model.add(Dense(OUTPUT_SIZE))
    # merged_model.add(PReLU())
    # merged_model.add(Dropout(DROPOUT_PROB))
    # merged_model.add(BatchNormalization())
    # merged_model.add(Dense(OUTPUT_SIZE))
    # merged_model.add(PReLU())
    # merged_model.add(Dropout(DROPOUT_PROB))
    # merged_model.add(BatchNormalization())
    # merged_model.add(Dense(OUTPUT_SIZE))
    # merged_model.add(PReLU())
    # merged_model.add(Dropout(DROPOUT_PROB))
    # merged_model.add(BatchNormalization())
    # merged_model.add(Dense(OUTPUT_SIZE))
    # merged_model.add(PReLU())
    merged_model.add(Dropout(DROPOUT_PROB))
    merged_model.add(BatchNormalization())
    merged_model.add(Dense(1))
    merged_model.add(Activation('sigmoid'))

    if weights_path is not None:
        print 'Restoring weights ... '
        merged_model.load_weights(weights_path)

    merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    merged_model.summary()
    return merged_model


def make_submission(test_proba):
    submission = pd.read_csv('../../data/sample_submission.csv')
    defaults = submission.is_duplicate.values.astype(float)
    defaults = defaults.ravel()

    for idx, (old, new) in enumerate(itertools.izip_longest(defaults, test_proba, fillvalue='0.356')):
        if new == '0.5':
            new = random.uniform(0.000001, 0.9999999)
        defaults[idx] = float(new)

    submission['is_duplicate'] = pd.Series(defaults, index=submission.index).astype(float)
    import time

    submission.to_csv('../../submissions/submission_{0}.csv'.format(time.time()), index=False)


def main():
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

    x1 = tk.texts_to_sequences(data.question1.values)
    x1 = sequence.pad_sequences(x1, maxlen=MAX_LEN)

    x2 = tk.texts_to_sequences(data.question2.values.astype(str))
    x2 = sequence.pad_sequences(x2, maxlen=MAX_LEN)

    word_index = tk.word_index

    if TRAIN_MODE:
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

        merged_model = make_monster_model(embedding_matrix, word_index, train_embeddings=False)
        checkpoint = ModelCheckpoint(WEIGHTS_NAME, monitor='val_loss', save_best_only=True, verbose=2)
        earlystop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0)
        input_arrays = [x1, x2, x1, x2]
        merged_model.fit(input_arrays, y=y, batch_size=BATCH_SIZE, nb_epoch=NUM_EPOCHS,
                         verbose=1, validation_split=0.2, shuffle=True, callbacks=[earlystop, checkpoint])

        train_proba = merged_model.predict(input_arrays, batch_size=BATCH_SIZE)
        cPickle.dump(train_proba.ravel(), open(TRAIN_OUTPUT_PROB_NAME, 'wb'), -1)
        merged_model.save(COMPILED_MODEL_NAME)
    else:
        Q1_test = pd.read_csv('../../data/test_question1_stemmed.csv')
        Q2_test = pd.read_csv('../../data/test_question2_stemmed.csv')

        data_test = pd.DataFrame()
        data_test['question1'] = Q1_test['0'].astype(str).values
        data_test['question2'] = Q2_test['0'].astype(str).values

        x1 = tk.texts_to_sequences(data_test.question1.values.astype(str))
        x1 = sequence.pad_sequences(x1, maxlen=MAX_LEN)
        x2 = tk.texts_to_sequences(data_test.question2.values.astype(str))
        x2 = sequence.pad_sequences(x2, maxlen=MAX_LEN)

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

        trained_model = make_monster_model(embedding_matrix, WEIGHTS_NAME)
        test_proba = trained_model.predict([x1, x2, x1, x2], batch_size=BATCH_SIZE)
        test_proba = test_proba.ravel()
        cPickle.dump(test_proba, open(TEST_OUTPUT_PROB_NAME, 'wb'), -1)


if __name__ == '__main__':
    main()
