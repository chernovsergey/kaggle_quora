import os

from sklearn.model_selection import StratifiedKFold

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.optimizers import Adamax, Nadam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from models.abshishek.prepare_datasets import prepare_train, prepare_test
from sklearn.metrics import log_loss

INPUT_FOLDER = '../../data'


class KerasMLPClassifier:
    def __init__(self, input_shape):
        model = Sequential()
        model.add(Dense(256, activation='tanh', input_shape=(input_shape,)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        optimizer = Nadam(lr=0.0005)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=['acc'])

        self.model = model
        self.earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        self.class_weight = {0: 1.309028344, 1: 0.472001959}

    def fit(self, X_train, y_train):
        print 'Fitting ... '
        self.history = self.model.fit(X_train, y_train,
                                      batch_size=512 * 2, epochs=150, verbose=1, shuffle=True,
                                      validation_split=0.2, callbacks=[self.earlystop], class_weight=self.class_weight)

    def predict(self, X_eval):
        preds = self.model.predict(X_eval, batch_size=2048, verbose=1)
        for _ in range(4):
            preds += self.model.predict(X_eval, batch_size=2048, verbose=1)
        preds /= 5.

        return preds


XTrain, features = prepare_train()
XTrain.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
XTrain.fillna(value=0, inplace=True)

XTest = prepare_test()
XTest.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
XTest.fillna(value=0, inplace=True)

YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values.ravel()
XTest = XTest.as_matrix()
XTrain = XTrain.as_matrix()

N_FOLDS = 10
skf = StratifiedKFold(n_splits=N_FOLDS).split(XTrain, YTrain)
splits = list(skf)

print "Creating train and test sets for blending."

preds_train = np.zeros((XTrain.shape[0],))
loglosses = []

preds_test_j = np.zeros((XTest.shape[0], N_FOLDS))
for fold_id, (train_indexes, predict_indexes) in enumerate(splits):
    print "Fold", fold_id

    clf = KerasMLPClassifier(XTrain.shape[1])
    clf.fit(XTrain[train_indexes],YTrain[train_indexes])

    y_pred = clf.predict(XTrain[predict_indexes])
    preds_train[predict_indexes] = y_pred

    lloss = log_loss(YTrain[predict_indexes], y_pred)
    loglosses.append(lloss)
    print 'LogLoss: ', lloss

    # Predict on entire test set
    preds_test_j[:, fold_id] = clf.predict(XTest).ravel()

print "Out of fold logloss-es:\n", loglosses

# Save OOF predictions as features
pd.DataFrame(preds_train, columns=['MLP_oof']).to_csv(INPUT_FOLDER + '/features/mlp_oof_train.csv')

# Save mean of predictions for test
preds_test = preds_test_j.mean(1)
pd.DataFrame(preds_test, columns=['MLP_oof']).to_csv(INPUT_FOLDER + '/features/mlp_oof_test.csv')
