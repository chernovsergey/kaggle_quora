import os
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


def mlp(X_train, X_eval, y_train, y_eval=None):
    model = Sequential()
    model.add(Dense(256, activation='tanh', input_shape=(X_train.shape[1],)))
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

    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

    if y_eval is not None:
        history = model.fit(X_train, y_train,
                            batch_size=2048, epochs=150, verbose=1, shuffle=True,
                            validation_data=(X_eval, y_eval), callbacks=[earlystop])
    else:
        class_weight = {0: 1.309028344, 1: 0.472001959}
        history = model.fit(X_train, y_train,
                            batch_size=512, epochs=150, verbose=1, shuffle=True,
                            validation_split=0.2, callbacks=[earlystop], class_weight=class_weight)

    preds = model.predict(X_eval, batch_size=2048, verbose=1)
    print 0
    for _ in range(4):
        preds += model.predict(X_eval, batch_size=2048, verbose=1)
    preds /= 5.

    bst_val_score = min(history.history['val_loss'])
    return preds, bst_val_score


train, features = prepare_train()
y = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values.ravel().tolist()
train.replace(to_replace=[np.inf, -np.inf], value=0, inplace=True)
train.fillna(value=0, inplace=True)
test = prepare_test()
preds, bst_eval_score = mlp(train.as_matrix(), test.as_matrix(), y)

test_ids = pd.read_csv(INPUT_FOLDER + '/sample_submission.csv', usecols=['test_id']).values.ravel().tolist()
submission_name = 'MLP_%.4f_' % (bst_eval_score) + '.csv'
print 'Saving to ', submission_name
submission = pd.DataFrame()
submission['test_id'] = pd.Series(test_ids)
submission['is_duplicate'] = pd.Series(preds.ravel())
submission.to_csv(submission_name, index=False)
