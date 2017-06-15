import os

from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.engine import Model
from keras.engine import Input

from sklearn.metrics import log_loss

from keras.layers import LSTM, concatenate
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Embedding
import pandas as pd

INPUT_FOLDER = '../../data'
TRAIN_MODE = True
NUM_EPOCHS = 2
OUTPUT_SIZE = 128
MAX_FEATURES = 200000
EMBEDDING_SIZE = 300
DROPOUT_PROB = 0.5
MAX_LEN = 20
BATCH_SIZE = 64
PATIENCE = 3
WEIGHTS_NAME = 'weights_convnet.h5'
COMPILED_MODEL_NAME = 'compiled_convnet_model.h5'
TRAIN_OUTPUT_PROB_NAME = '../../data/train_convnet_proba.pkl'
GLOVE_VECTORDS_PATH = '../../data/glove.840B.300d.txt'
INPUT_TRAIN_FILE = '../../data/train.csv'
INPUT_TEST_FILE = '../../data/test.csv'
TEST_OUTPUT_PROB_NAME = '../../data/test_convnet_proba.pkl'
BASE_DIR = '../../data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin.gz'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'

import cPickle as pickle


class KerasConvnet:
    def __init__(self, use_weights=None):
        self.make_model(use_weights)
        pass

    def make_model(self, use_weights):
        num_lstm = 267
        num_dense = 117
        rate_drop_lstm = 0.25
        rate_drop_dense = 0.15

        act = 'relu'

        embedding_layer = Embedding(nb_words,
                                    EMBEDDING_SIZE,
                                    weights=[embedding_matrix],
                                    input_length=MAX_LEN,
                                    trainable=False)
        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        sequence_1_input = Input(shape=(MAX_LEN,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(MAX_LEN,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        y1 = lstm_layer(embedded_sequences_2)

        merged = concatenate([x1, y1])
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input], outputs=preds)

        if use_weights is not None:
            model.load_weights(use_weights)

        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['acc'])
        model.summary()

        self.model = model
        self.earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        self.checkpoint = ModelCheckpoint(WEIGHTS_NAME, monitor='val_loss', save_best_only=True, verbose=2)
        self.class_weight = {0: 1.309028344, 1: 0.472001959}

    def fit(self, train1, train2, y_train, valid1, valid2, y_valid):
        print 'Fitting ... '
        input_arrays = [train1, train2]

        weight_val = np.ones(len(y_valid))
        weight_val *= 0.472001959
        weight_val[y_valid == 0] = 1.309028344

        self.model.fit(input_arrays, y_train, batch_size=BATCH_SIZE,
                       epochs=NUM_EPOCHS, validation_data=([valid1, valid2], y_valid, weight_val), shuffle=True,
                       callbacks=[self.earlystop, self.checkpoint])

    def predict(self, test1, test2):
        input_arrays = [test1, test2]
        preds = self.model.predict(input_arrays, batch_size=BATCH_SIZE, verbose=1)
        for _ in range(0):
            preds += self.model.predict(input_arrays, batch_size=BATCH_SIZE, verbose=1)
        preds /= 1.

        return preds


YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values.ravel()

ids = pd.read_csv('data_ids.csv', usecols=['graph_id'])
graph_ids_unique = ids.graph_id.unique()

with open('dump_sequences.pkl', 'rb') as f:
    x1_train, x2_train, x1_test, x2_test, nb_words = pickle.load(f)
embedding_matrix = np.load('embedding_matrix.npy')

print x1_train.shape, x2_train.shape
print x1_test.shape, x2_test.shape
print nb_words

N_FOLDS = 5
N_ITER = 3

print "Creating train and test sets for blending."
preds_train = np.zeros((x1_train.shape[0]))
preds_test_j = np.zeros((x1_test.shape[0]))
loglosses = []

from sklearn.model_selection import KFold

for iter_num in range(N_ITER):
    kf = KFold(n_splits=N_FOLDS, shuffle=True)
    splits = list(kf.split(graph_ids_unique))

    for fold_id, (train_graphs, test_graphs) in enumerate(splits):
        print "Fold", fold_id
        train_ind = ids[ids.graph_id.isin(graph_ids_unique[train_graphs])].index.values  # temporary train indexes
        test_ind = ids[ids.graph_id.isin(graph_ids_unique[test_graphs])].index.values  # final test indexes

        kf_valid = KFold(n_splits=8)
        graph_ids_train = graph_ids_unique[train_graphs]
        train_graphs, valid_graphs = list(kf_valid.split(graph_ids_train))[0]
        train_ind = ids[ids.graph_id.isin(graph_ids_train[train_graphs])].index.values  # final train indexes
        valid_ind = ids[ids.graph_id.isin(graph_ids_train[valid_graphs])].index.values  # final valid indexes

        clf = KerasConvnet()
        clf.fit(x1_train[train_ind][:1000], x2_train[train_ind][:1000], YTrain[train_ind][:1000],
                x1_train[valid_ind][:1000], x2_train[valid_ind][:1000], YTrain[valid_ind][:1000])

        y_pred = clf.predict(x1_train[test_ind], x2_train[test_ind])
        preds_train[test_ind] += y_pred.ravel()

        lloss = log_loss(YTrain[test_ind], y_pred)
        loglosses.append(lloss)
        print 'LogLoss: ', lloss

        # Predict on entire test set
        preds_test_j += clf.predict(x1_test, x2_test).ravel()

    print "Out of fold logloss-es:\n", loglosses
    print 'train score:', log_loss(YTrain, preds_train / (iter_num + 1))

preds_train /= N_ITER
preds_test_j /= (N_ITER * N_FOLDS)

# Save OOF predictions as features
pd.DataFrame(preds_train, columns=['siamese_oof']).to_csv(INPUT_FOLDER + '/features/siamese_oof_train.csv')

# Save mean of predictions for test
preds_test = preds_test_j.mean(1)
pd.DataFrame(preds_test, columns=['siamese_oof']).to_csv(INPUT_FOLDER + '/features/siamese_oof_test.csv')
