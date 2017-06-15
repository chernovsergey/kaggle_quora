from __future__ import division

import time

from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics.classification import log_loss
from sklearn.metrics.ranking import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pylab as plt
from xgboost import plot_importance

from models.abshishek.prepare_datasets import load_sparse_csr

features = []

INPUT_FOLDER = '../../data'
path = '../input/'

drop_list = []
MODELS_DIR = './stacking/'
FEATURES_DIR = './stacking/features'


def prepare_train():
    df_train = pd.read_csv('train_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/trainftr.csv', usecols=['train'])
    df_train['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_train.csv', index_col=0)
    df_train = df_train.join(mephistoph)

    return df_train


def prepare_test():
    df_test = pd.read_csv('test_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/testftr.csv', usecols=['test'])
    df_test['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_test.csv', index_col=0)
    df_test = df_test.join(mephistoph)

    return df_test

def modifier_magic(y, beta=0.5):
    return 0.5 * ((2 * abs(y - 0.5)) ** beta) * np.sign(y - 0.5) + 0.5


def modifier_stat_efficient(x, seed_a=0.165, seed_b=0.370):
    #  a * x / (a * x + b * (1 - x))
    a = seed_a / seed_b
    b = (1.0 - seed_a) / (1.0 - seed_b)
    result = a * x / (a * x + b * (1.0 - x))
    return result


def calibrate(name, modifier):
    df = pd.read_csv(name, index_col=0)
    result = []
    for val in df.ix[:, 0].values[:]:
        new_val = modifier(val)
        result.append(new_val)

    df['is_duplicate'] = np.array(result)
    df.to_csv(name.split('.')[0] + '_calibrated.csv', index=True)


def make_submission_name(base_name):
    return base_name + str(time.time()) + '.csv'


class LightGBMWrapper():
    def __init__(self, params):
        self.params_ = params

    def fit(self, X, y):
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.15, stratify=y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.bst = lgb.train(self.params_, lgb_train,
                             num_boost_round=self.params_['n_estimators'],
                             valid_sets=[lgb_train, lgb_eval],
                             verbose_eval=self.params_['verbose_eval'],
                             early_stopping_rounds=self.params_['early_stopping_rounds'])

    def predict(self, X):
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)


params_lgb = {
    'verbose': 0,
    'verbose_eval': 100,
    'task': 'train', 'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'num_leaves': 2 ** 7,
    # 'learning_rate': 0.05,
    'feature_fraction': 0.725,
    'bagging_fraction': 0.725,
    # 'min_child_weight': 2,
    'bagging_freq': 5,
    'n_estimators': 3000,
    'early_stopping_rounds': 50,
    'n_folds': 5,
    'seed': np.random.randint(7, 30) + np.random.randint(0, 100)
}

if __name__ == '__main__':

    RANDOM_STATE = 542
    N_FOLDS = 5

    # Right way
    XTrain = load_sparse_csr('tfidf_stem_train.npz')# prepare_train()
    XTrain = XTrain.tocsr()
    # XTrain.fillna(0, inplace=True)
    # XTrain.replace([np.inf, -np.inf], 0, inplace=True)

    YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values

    XTest = load_sparse_csr('tfidf_stem_test.npz')# prepare_test()
    # XTest.fillna(0, inplace=True)
    # XTest.replace([np.inf, -np.inf], 0, inplace=True)

    # features = XTrain.columns
    # print len(features), ' features total'
    # print "\n".join(features)

    # XTrain = XTrain.ix[:, features].as_matrix()
    YTrain = np.array(YTrain).ravel()

    # XTest = XTest.ix[:, features].as_matrix()

    skf = StratifiedKFold(n_splits=N_FOLDS).split(np.zeros((XTrain.shape[0],1)), np.zeros(len(YTrain)))
    splits = list(skf)


    base_classifiers = [LightGBMWrapper(params_lgb)]  #RidgeClassifier(alpha=0.05)

    print "Creating train and test sets for blending."

    df_blend_train = np.zeros((XTrain.shape[0], len(base_classifiers)))
    df_blend_test = np.zeros((XTest.shape[0], len(base_classifiers)))
    oof_loglosses = np.zeros((len(base_classifiers), len(splits)))

    for clf_id, clf in enumerate(base_classifiers):

        print "Training base classifier #{0} -- {1}".format(clf_id, clf.__class__.__name__)

        dataset_blend_test_j = np.zeros((XTest.shape[0], N_FOLDS))
        for fold_id, (train_indexes, predict_indexes) in enumerate(splits):
            print "Fold", fold_id

            # Fit on train part
            clf.fit(XTrain[train_indexes], YTrain[train_indexes])

            # Predict on the rest of data
            y_pred = clf.predict(XTrain[predict_indexes])
            df_blend_train[predict_indexes, clf_id] = y_pred
            lloss = log_loss(YTrain[predict_indexes], y_pred)
            oof_loglosses[clf_id, fold_id] = lloss
            print 'LogLoss: ', lloss

            # Predict on entire test set
            y_pred_test = clf.predict(XTest)
            # y_pred_test = modifier_stat_efficient(y_pred_test)
            dataset_blend_test_j[:, fold_id] = y_pred_test

        # Average predictions for test set
        df_blend_test[:, clf_id] = dataset_blend_test_j.mean(1)

    print "Out of fold logloss-es:\n", oof_loglosses

    np.save('boost_sprasetfidf_train.csv', df_blend_train)
    np.save('boost_sparsetfidf_test.csv', df_blend_test)