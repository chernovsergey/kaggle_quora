from __future__ import division

import os
import time
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pylab as plt
from xgboost import plot_importance
import itertools
import cPickle as pickle

features = []

INPUT_FOLDER = '../../data'

path = '../input/'

drop_list = []
MODELS_DIR = './stacking/'
FEATURES_DIR = './stacking/features'


def prepare_train():
    df_train = pd.read_csv(INPUT_FOLDER + '/features/all_features_train_17april.csv', index_col=0)
    return df_train


def prepare_test():
    df_test = pd.read_csv(INPUT_FOLDER + '/features/all_features_test_17april.csv', index_col=0)
    return df_test


class XGBoostWrapper():
    def __init__(self, params):
        self.params_ = params
        self.features = params['features']

    def fit(self, X, y):
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, stratify=y)
        d_train = xgb.DMatrix(X_train, label=y_train, feature_names=self.features)
        d_valid = xgb.DMatrix(X_eval, label=y_eval, feature_names=self.features)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        self.bst = xgb.train(self.params_, d_train, evals=watchlist,
                             num_boost_round=self.params_['n_estimators'],
                             early_stopping_rounds=self.params_['early_stopping_rounds'],
                             verbose_eval=100)

    def predict(self, X):
        d_test = xgb.DMatrix(X, feature_names=self.features)
        return self.bst.predict(d_test)  # num_iteration=self.bst.best_iteration)


class LightGBMWrapper():
    def __init__(self, params):
        self.params_ = params

    def fit(self, X, y):
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, stratify=y)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.bst = lgb.train(self.params_, lgb_train,
                             num_boost_round=self.params_['n_estimators'],
                             valid_sets=[lgb_train, lgb_eval],
                             verbose_eval=self.params_['verbose_eval'],
                             early_stopping_rounds=self.params_['early_stopping_rounds'])

        return self.bst

    def predict(self, X):
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)


def modifier_magic(y, beta=0.5):
    return 0.5 * ((2 * abs(y - 0.5)) ** beta) * np.sign(y - 0.5) + 0.5


def modifier_stat_efficient(x, seed_a=0.165, seed_b=0.370):
    #  a * x / (a * x + b * (1 - x))
    a = seed_a / seed_b
    b = (1 - seed_a) / (1 - seed_b)
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


def train_stacking_level(level_name, XTrain, YTrain, XTest,
                         seed, num_folds, level_classifiers, feature_names):
    """

    :param level_name: will be used as prefix at data saving time
    :param XTrain: train dataset
    :param YTrain: target values
    :param XTest: test dataset
    :param seed: level seed used for stratified k-fold
    :param num_folds: number of folds
    :param level_classifiers: list of classifiers
    :return: None, but saves extended versions of XTrain and XTest with prefix :param level_name
    """

    print 10 * '===', level_name.upper(), 10 * '==='

    skf = StratifiedKFold(n_splits=num_folds, random_state=seed).split(XTrain, YTrain)
    splits = list(skf)

    df_blend_train = np.zeros((XTrain.shape[0], len(level_classifiers)))
    df_blend_test = np.zeros((XTest.shape[0], len(level_classifiers)))
    oof_loglosses = np.zeros((len(level_classifiers), len(splits)))

    for clf_id, clf in enumerate(level_classifiers):

        print "Level: {0} Base classifier #{1} -- {2}".format(level_name.upper(), clf_id, clf.__class__.__name__)

        dataset_blend_test_j = np.zeros((XTest.shape[0], num_folds))
        for fold_id, (train_id, predict_id) in enumerate(splits):
            print "Fold", fold_id, "out of", num_folds
            print 'train:', min(train_id), max(train_id), 'test:', min(predict_id), max(predict_id)
            # Fit on train part
            model = clf.fit(XTrain[train_id], YTrain[train_id])
            print 'Saving to', modelname, '...'
            with open(modelname, 'wb') as fout:
                pickle.dump(gbm, fout)

            # Predict on the rest of data
            y_pred = clf.predict(XTrain[predict_id])
            df_blend_train[predict_id, clf_id] = y_pred
            lloss = log_loss(YTrain[predict_id], y_pred)
            oof_loglosses[clf_id, fold_id] = lloss
            print 'OOF LogLoss: ', lloss

            # Predict on entire test set
            dataset_blend_test_j[:, fold_id] = clf.predict(XTest)

        # Average predictions for test set
        df_blend_test[:, clf_id] = dataset_blend_test_j.mean(1)

    print "Out of fold logloss-es:\n", oof_loglosses

    new_feature_names = ["{0}_{1}".format(level_name, i) for i in range(len(level_classifiers))]
    # new_feature_names = list(itertools.chain(new_feature_names, feature_names))

    # extended_train = np.concatenate((df_blend_train, XTrain), axis=1)
    # print 'Extended train set shape:', extended_train.shape

    # extended_test = np.concatenate((df_blend_test, XTest), axis=1)
    # print 'Extended test set shape:', extended_test.shape

    savedir = FEATURES_DIR + '/' + level_name
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    np.save(savedir + '/splits_' + level_name, np.array(splits).reshape((-1, 2)))
    pd.DataFrame(extended_train, columns=new_feature_names).to_csv(savedir + '/extended_train_set.csv')
    pd.DataFrame(extended_test, columns=new_feature_names).to_csv(savedir + '/extended_test_set.csv')


if __name__ == '__main__':
    RANDOM_STATE_LEVEL = 542
    RANDOM_STATE_FINAL_SPLIT = 629
    N_FOLDS = 3
    SUBMISSION_NAME = make_submission_name('submit_GBM_stacking')

    params_xgb2 = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],
        'features': features,
        'colsample_bytree': 0.925,
        'max_depth': 10,
        'n_estimators': 500,
        'min_child_weight': 50,
        'early_stopping_rounds': 50,
        # 'lambda': 2.15,
        # 'alpha': 0.9,
        'seed': np.random.randint(13, 30) + np.random.randint(0, 100),
        'silent': 1}

    params_xgb1 = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss'],
        'features': features,
        # 'subsample': 1.0,
        'colsample_bytree': 0.925,
        'max_depth': 5,
        'n_estimators': 1000,
        'min_child_weight': 2,
        'early_stopping_rounds': 50,
        'seed': np.random.randint(11, 30) + np.random.randint(0, 100),
        'silent': 1,
    }


    def level_1(name):
        # Right way
        XTrain = prepare_train()
        YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values
        XTest = prepare_test()
        XTrain, YTrain = shuffle(XTrain, YTrain)

        # With hold-out for debugging
        # X = prepare_train()
        # y = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values
        # XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE_LEVEL,
        #                                                 stratify=y)
        # XTrain, YTrain = shuffle(XTrain, YTrain)

        features = set(XTrain.columns).intersection(set(XTest.columns))
        features = filter(lambda x: 'porter_intera' not in x, features)
        print len(features), ' features total'
        print "\n".join(features)

        XTrain = XTrain.ix[:, features].as_matrix()
        YTrain = np.array(YTrain).ravel()

        XTest = XTest.ix[:, features].as_matrix()
        # YTest = np.array(YTest).ravel()

        params_lgb1 = {
            'verbose': 0,
            'verbose_eval': 100,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            # 'feature_fraction': 0.725,  # 0.57,
            'bagging_fraction': 0.925,
            'min_child_weight': 2,
            'bagging_freq': 5,
            'n_estimators': 1500,
            'early_stopping_rounds': 50,
            'max_depth': 5,
            'seed': np.random.randint(7, 30) + np.random.randint(0, 100)
        }

        params_lgb2 = {
            'verbose': 0,
            'verbose_eval': 100,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            'bagging_fraction': 0.925,
            # 'min_child_weight': 2,
            'bagging_freq': 5,
            'n_estimators': 1500,
            'early_stopping_rounds': 50,
            'max_depth': 7,
            'seed': np.random.randint(7, 30) + np.random.randint(0, 100)
        }

        params_lgb3 = {
            'verbose': 0,
            'verbose_eval': 100,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            'bagging_fraction': 0.925,
            'min_child_weight': 50,
            # 'min_child_samples': 15,
            'bagging_freq': 15,
            'n_estimators': 3500,
            'early_stopping_rounds': 50,
            'max_depth': 10,
            # 'reg_alpha': 2.75,
            # 'reg_lambda': 1.5,
            'seed': np.random.randint(11, 30) + np.random.randint(0, 100)
        }

        params_lgb4 = {
            'verbose': 0,
            'verbose_eval': 100,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            'bagging_fraction': 0.925,
            'min_child_weight': 20,
            # 'min_child_samples': 15,
            'bagging_freq': 15,
            'n_estimators': 3500,
            'early_stopping_rounds': 50,
            'max_depth': 8,
            # 'reg_alpha': 2.75,
            # 'reg_lambda': 1.5,
            'seed': np.random.randint(11, 30) + np.random.randint(0, 100)
        }

        level_clf = [LightGBMWrapper(params_lgb1),  # LightGBMWrapper(params_lgb2),
                     LightGBMWrapper(params_lgb3)]  # LightGBMWrapper(params_lgb4)]

        train_stacking_level(name, XTrain, YTrain, XTest, RANDOM_STATE_LEVEL, N_FOLDS, level_clf, features)


    def level_2(name):
        XTrain = pd.read_csv(FEATURES_DIR + '/TEST_LEVEL/extended_train_set.csv', index_col=0)
        features = XTrain.columns
        XTrain = XTrain.as_matrix()
        YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values[:].ravel()
        XTest = pd.read_csv(FEATURES_DIR + '/TEST_LEVEL/extended_test_set.csv', index_col=0).as_matrix()

        params_lgb1 = {
            'verbose': 0,
            'verbose_eval': 5,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            # 'feature_fraction': 0.725,  # 0.57,
            # 'bagging_fraction': 0.925,
            # 'min_child_weight': 2,
            'bagging_freq': 5,
            'n_estimators': 1500,
            'early_stopping_rounds': 50,
            'max_depth': 2,
            'seed': np.random.randint(7, 30) + np.random.randint(0, 100)
        }

        params_lgb2 = {
            'verbose': 0,
            'verbose_eval': 100,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss'],
            # 'bagging_fraction': 0.925,
            # 'min_child_weight': 20,
            # 'min_child_samples': 15,
            'bagging_freq': 5,
            'n_estimators': 3500,
            'early_stopping_rounds': 50,
            'max_depth': 3,
            'seed': np.random.randint(11, 30) + np.random.randint(0, 100)
        }

        level_clf = [LightGBMWrapper(params_lgb1), LightGBMWrapper(params_lgb2)]

        train_stacking_level(name, XTrain, YTrain, XTest, RANDOM_STATE_LEVEL, 5, level_clf, features)


    level_1('test_level')
    # level_2('test_level_2')

    XTrain = pd.read_csv(FEATURES_DIR + '/test_level/extended_train_set.csv', index_col=0)
    features = XTrain.columns
    XTrain = XTrain.as_matrix()
    YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values[:].ravel()
    XTest = pd.read_csv(FEATURES_DIR + '/test_level/extended_test_set.csv', index_col=0).as_matrix()
    XTrain, YTrain = shuffle(XTrain, YTrain)

    params_final_lgb = {
        'boosting_type': 'gbdt', 'objective': 'binary', 'metric': ['binary_logloss'], 'n_estimators': 500,
        'max_depth': 4,
        # 'feature_fraction': 0.725,
        'bagging_fraction': 0.925,
        'bagging_freq': 5, 'verbose': 0,
        'seed': np.random.randint(7, 30),
        # 'reg_alpha': 2.75 ** 2, 'reg_lambda': 1.5,
        # 'learning_rate': 0.05,
        'early_stopping_rounds': 25,
    }

    X_final, X_final_eval, y_final, y_final_eval = train_test_split(XTrain, YTrain, test_size=0.33,
                                                                    random_state=RANDOM_STATE_LEVEL, stratify=YTrain)
    lgb_final = lgb.Dataset(X_final, y_final)
    lgb_final_eval = lgb.Dataset(X_final_eval, y_final_eval)
    gbm = lgb.train(params_final_lgb, lgb_final,
                    num_boost_round=params_final_lgb['n_estimators'],
                    valid_sets=[lgb_final, lgb_final_eval],
                    verbose_eval=10,
                    early_stopping_rounds=params_final_lgb['early_stopping_rounds'],
                    feature_name=list(features))

    y_submission = gbm.predict(XTest)
    # print log_loss(YTest, y_submission)

    lgb.plot_importance(gbm)

    print 'Submission saved to', SUBMISSION_NAME
    plt.show()

    print 'Predicting ... '
    sample_submission = pd.read_csv(INPUT_FOLDER + '/sample_submission.csv')
    test_id = sample_submission.ix[:, 0]
    submit = pd.DataFrame()
    submit['test_id'] = test_id
    submit['is_duplicate'] = y_submission
    header = ["test_id", "is_duplicate"]
    submit.to_csv(SUBMISSION_NAME, columns=header, index=False)

    print 'Calibrating predictions ... '
    calibrate(SUBMISSION_NAME, modifier=modifier_stat_efficient)
