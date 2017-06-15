# coding: utf-8
import os
import time
import xgbfir
from scipy.sparse import csr_matrix
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.classification import log_loss

from models.abshishek.prepare_datasets import prepare_train, prepare_test, load_sparse_csr

os.environ["KERAS_BACKEND"] = "tensorflow"
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection._split import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import seaborn as sns
from scipy import sparse
import cPickle as pickle

import datetime

mydate = datetime.datetime.now()
TODAY = str(mydate.today().date())[-2:] + '_' + str(mydate.today().month)
INPUT_FOLDER = '../../data'
MODELS_FOLDER = './models'


def sumbmission_corr(file1, file2):
    df1 = pd.read_csv(file1, index_col=0).values.ravel()
    df2 = pd.read_csv(file2, index_col=0).values.ravel()
    import scipy.stats as st
    print st.spearmanr(df1, df2)[0], 'spearman'
    print st.pearsonr(df1, df2)[0], 'pearson'


def mix(fname1, fname2, w1, w2, output_name):
    assert w1 + w2 == 1.0
    import pandas as pd
    p1 = pd.read_csv(fname1, usecols=['is_duplicate'])
    p2 = pd.read_csv(fname2, usecols=['is_duplicate'])
    df = pd.read_csv(INPUT_FOLDER + '/sample_submission.csv')
    df['one'] = p1
    df['two'] = p2
    df['is_duplicate'] = df.apply(lambda x: w1 * x['one'] + w2 * x['two'], axis=1)
    df[['test_id', 'is_duplicate']].to_csv(output_name, index=False)


def modifier_stat_efficient(x, seed_a=0.165, seed_b=0.370):
    #  a * x / (a * x + b * (1 - x))
    a = seed_a / seed_b
    b = (1.0 - seed_a) / (1.0 - seed_b)
    result = a * x / (a * x + b * (1.0 - x))
    return result


def modifier_magic(y, beta=0.5):
    return 0.5 * ((2 * abs(y - 0.5)) ** beta) * np.sign(y - 0.5) + 0.5


def calibrate(name, modifier):
    print 'Calibrating', name
    df = pd.read_csv(name, index_col=0)
    result = []
    for val in df.ix[:, 0].values[:]:
        new_val = modifier(val)
        result.append(new_val)

    df['is_duplicate'] = np.array(result)
    df.to_csv(name.split('.')[0] + '_calibrated.csv', index=True)


def oversample(features, labels, p=0.165):
    labels = np.array(labels)
    pos_rows = np.where(labels == 1)
    neg_rows = np.where(labels == 0)
    pos_train = features[pos_rows]
    neg_train = features[neg_rows]

    # p = 0.165  # 0.1542
    scale = ((float(len(pos_train)) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = np.concatenate([neg_train, neg_train], axis=0)
        scale -= 1
    neg_train = np.concatenate([neg_train, neg_train[:int(scale * len(neg_train))]], axis=0)

    print float(len(pos_train)) / (len(pos_train) + len(neg_train))

    df_train_rebalanced = np.concatenate([pos_train, neg_train], axis=0)
    y_rebalanced = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train
    print "rebalance done"

    return df_train_rebalanced, y_rebalanced


def oversample_sparse(features, labels, p=0.165):
    labels = np.array(labels)
    pos_rows = np.where(labels == 1)
    neg_rows = np.where(labels == 0)
    pos_train = features[pos_rows]
    neg_train = features[neg_rows]
    print pos_train.shape
    print neg_train.shape

    pos_len = len(pos_rows[0])
    neg_len = len(neg_rows[0])

    # p = 0.165  # 0.1542
    scale = ((float(pos_len) / (pos_len + neg_len)) / p) - 1
    while scale > 1:
        neg_train = sparse.vstack([neg_train, neg_train])
        scale -= 1
    neg_train = sparse.vstack([neg_train, neg_train[:int(scale * neg_train.shape[0])]])

    print float(pos_len) / (pos_len + neg_train.shape[0])

    df_train_rebalanced = sparse.vstack([pos_train, neg_train])
    y_rebalanced = (np.zeros(pos_len) + 1).tolist() + np.zeros(neg_train.shape[0]).tolist()
    del pos_train, neg_train
    print "rebalance done"

    print df_train_rebalanced.shape, len(y_rebalanced)
    return df_train_rebalanced, y_rebalanced


def make_submission(name, y_pred):
    sample_submission = pd.read_csv(INPUT_FOLDER + '/sample_submission.csv')
    test_id = sample_submission.ix[:, 0]
    submit = pd.DataFrame()
    submit['test_id'] = test_id
    submit['is_duplicate'] = y_pred
    header = ["test_id", "is_duplicate"]
    submit.to_csv(name, columns=header, index=False)


class LightGBMBooster:
    def __init__(self, params, seed, modelname='', show_importance=True):
        self.params = params
        self.show_importance = show_importance
        self.constant_seed = seed

        if modelname == '':
            self.modelname = 'lightgbm_{0}_seed={1}_valloss.pkl'.format(TODAY, seed)
        else:
            self.modelname = modelname

    def fit(self, X_train, X_eval, y_train, y_eval, feature_names):
        print self.params
        print 'LightGBM: training ... '
        eval_result = {}
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.gbm = lgb.train(self.params, lgb_train,
                             num_boost_round=self.params['n_estimators'],
                             valid_sets=[lgb_train, lgb_eval],
                             verbose_eval=self.params['verbose_eval'],
                             evals_result=eval_result,
                             early_stopping_rounds=self.params['early_stopping_rounds'],
                             feature_name=feature_names)

        valloss_best = str(eval_result['valid_1']['binary_logloss'][-1])
        modelname_new = self.modelname.split('.pkl')[0] + valloss_best + '.pkl'
        print 'Saving to', modelname_new, '...'
        with open(MODELS_FOLDER + '/' + modelname_new, 'wb') as fout:
            pickle.dump(self.gbm, fout)

        if self.show_importance:
            lgb.plot_importance(self.gbm, max_num_features=len(feature_names),
                                figsize=(150, 150), importance_type='gain')
            plt.show()

            # lgb.plot_importance(self.gbm, max_num_features=len(feature_names),
            #                     figsize=(150, 150), importance_type='split')
            # plt.show()
            #
            # lgb.plot_metric(eval_result, metric='binary_logloss')
            # plt.show()

        return self.gbm, eval_result, modelname_new

    def fit2(self, X, y, feature_names):
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, stratify=y, test_size=0.20,
                                                            random_state=np.random.randint(50, 1000))
        print self.params
        print 'LightGBM: training ... '
        eval_result = {}
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        self.gbm = lgb.train(self.params, lgb_train,
                             num_boost_round=self.params['n_estimators'],
                             valid_sets=[lgb_train, lgb_eval],
                             verbose_eval=self.params['verbose_eval'],
                             evals_result=eval_result,
                             early_stopping_rounds=self.params['early_stopping_rounds'],
                             feature_name=feature_names)

        return self.gbm, eval_result

    def fit_all(self, X, y, num_trees, feature_names):
        print self.params
        print 'LightGBM: training ... '
        lgb_train = lgb.Dataset(X, y)
        self.gbm = lgb.train(self.params, lgb_train,
                             num_boost_round=num_trees,
                             valid_sets=[lgb_train],
                             verbose_eval=self.params['verbose_eval'],
                             feature_name=feature_names)

        modelname_new = self.modelname.split('.pkl')[0] + 'all_data' + '.pkl'
        print 'Saving to', modelname_new, '...'
        with open(MODELS_FOLDER + '/' + modelname_new, 'wb') as fout:
            pickle.dump(self.gbm, fout)

        if self.show_importance:
            lgb.plot_importance(self.gbm, max_num_features=len(feature_names),
                                figsize=(150, 150), importance_type='gain')
            plt.show()

            lgb.plot_importance(self.gbm, max_num_features=len(feature_names),
                                figsize=(150, 150), importance_type='split')
            plt.show()

        return self.gbm, modelname_new

    def cv(self, X, y):
        print self.params
        print 'LightGBM: Cross validation with', self.params['n_folds'], 'folds'

        lgb_train = lgb.Dataset(X, y)
        lgb.cv(self.params, lgb_train,
               num_boost_round=self.params['n_estimators'],
               nfold=self.params['n_folds'],
               stratified=True,
               shuffle=True,
               early_stopping_rounds=self.params['early_stopping_rounds'],
               verbose_eval=self.params['verbose_eval'],
               callbacks=[lgb.callback.early_stopping(self.params['early_stopping_rounds'])],
               seed=self.constant_seed)

    def predict(self, X_test):
        print 'Predicting ... '
        y_pred = self.gbm.predict(X_test, num_iteration=self.gbm.best_iteration)
        return y_pred


class XGBooster:
    def __init__(self, params, seed, modelname='', show_importance=True):
        self.params = params
        self.show_importance = show_importance
        self.constant_seed = seed

        if modelname == '':
            self.modelname = 'xgboost_{0}_seed={1}_valloss.pkl'.format(TODAY, seed)
        else:
            self.modelname = modelname

    def fit(self, X_train, X_eval, y_train, y_eval, feature_names):
        print self.params
        print 'XGBoost: training ... '
        eval_result = {}
        d_train = xgb.DMatrix(X_train, label=y_train)
        d_valid = xgb.DMatrix(X_eval, label=y_eval)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        self.gbm = xgb.train(self.params, d_train,
                             evals=watchlist,
                             num_boost_round=self.params['n_estimators'],
                             early_stopping_rounds=self.params['early_stopping_rounds'],
                             verbose_eval=self.params['verbose_eval'],
                             evals_result=eval_result)

        try:
            xgbfir.saveXgbFI(self.gbm)
        except:
            pass

        valloss_best = str(eval_result['valid']['logloss'][-1])
        modelname_new = self.modelname.split('.pkl')[0] + valloss_best + '.pkl'
        print 'Saving to', modelname_new, '...'
        with open(MODELS_FOLDER + '/' + modelname_new, 'wb') as fout:
            pickle.dump(self.gbm, fout)

        if self.show_importance:
            xgb.plot_importance(self.gbm, importance_type='weight')
            plt.show()

            xgb.plot_importance(self.gbm, importance_type='gain')
            plt.show()

            xgb.plot_importance(self.gbm, importance_type='cover')
            plt.show()

        return self.gbm, eval_result, modelname_new

    def fit_all(self, X, y, num_trees, feature_names):
        print self.params
        print 'XGBoost: training ... '
        d_train = xgb.DMatrix(X, label=y)
        watchlist = [(d_train, 'train')]
        self.gbm = xgb.train(self.params, d_train,
                             evals=watchlist,
                             num_boost_round=num_trees,
                             verbose_eval=self.params['verbose_eval'])

        try:
            xgbfir.saveXgbFI(self.gbm)
        except:
            pass

        modelname_new = self.modelname.split('.pkl')[0] + 'alldata' + '.pkl'
        print 'Saving to', modelname_new, '...'
        with open(MODELS_FOLDER + '/' + modelname_new, 'wb') as fout:
            pickle.dump(self.gbm, fout)

        if self.show_importance:
            xgb.plot_importance(self.gbm, importance_type='weight')
            plt.show()

            xgb.plot_importance(self.gbm, importance_type='gain')
            plt.show()

            xgb.plot_importance(self.gbm, importance_type='cover')
            plt.show()

        return self.gbm, modelname_new

    def cv(self, X, y):
        print self.params
        print 'XGBooset: Cross validation with', self.params['n_folds'], 'folds'

        xgb_train = xgb.DMatrix(X, y)
        xgb.cv(self.params, xgb_train,
               num_boost_round=self.params['n_estimators'],
               nfold=self.params['n_folds'],
               stratified=True,
               early_stopping_rounds=self.params['early_stopping_rounds'],
               verbose_eval=self.params['verbose_eval'],
               seed=self.constant_seed)

    def predict(self, X_test):
        print 'Predicting ... '
        d_test = xgb.DMatrix(X_test)
        y_pred = self.gbm.predict(d_test, ntree_limit=self.gbm.best_iteration)
        return y_pred


params_light = {
    'verbose': 0,
    'verbose_eval': 100,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'num_leaves': 2 ** 7,
    'learning_rate': 0.025,
    'feature_fraction': 0.725,
    'bagging_fraction': 0.725,
    # 'min_child_weight': 2,
    'bagging_freq': 5,
    'n_estimators': 3000,
    'early_stopping_rounds': 50,
    'n_folds': 3,
    'seed': np.random.randint(7, 30) + np.random.randint(0, 100)
}

params_tough = {
    'verbose': 0,
    'verbose_eval': 100,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    # 'learning_rate': 0.0575,
    'feature_fraction': 0.725,
    'bagging_fraction': 0.725,
    # 'min_child_weight': 50,
    # 'min_child_samples': 15,
    'bagging_freq': 5,
    'n_estimators': 5000,
    'early_stopping_rounds': 50,
    'num_leaves': 2 ** 7,
    'max_depth': 7,
    'n_folds': 5,
    'oversample': False,
    # 'device': 'gpu',
    # 'reg_alpha': 2 ** 2,
    # 'reg_lambda': 2 ** 2,
    # 'lambda_l1': 2 ** 2,
    # 'lambda_l2': 2 ** 2,
    'seed': np.random.randint(7, 550) + np.random.randint(0, 100)
}

params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss'],
    'verbose_eval': 10,
    'learning_rate': 0.05,
    'subsample': 0.925,
    'colsample_bytree': 0.925,
    'max_depth': 7,
    'n_estimators': 1500,
    'early_stopping_rounds': 50,
    'n_folds': 3,
    'seed': np.random.randint(13, 30) + np.random.randint(0, 100),
    'silent': 1
}


def train_boost(booster, seed, oversampling=-1.0, use_tfidf=False, enable_cv=False, use_alldata=False, num_trees=-1):
    train, y, features = prepare_train()
    if use_tfidf:
        print 'Using raw tf-idf sparse matrix ... '
        features = 'auto'
        train_sparse = sparse.csr_matrix(train.values)
        # tfidf_sparse = load_sparse_csr('tfidf_stem_train.npz')
        bm25_sparse = load_sparse_csr('bm25_train.npz')
        # bm25_sparse = bm25_sparse[404290 - 50000:, :]
        # train = sparse.hstack([train_sparse, tfidf_sparse])
        # common_words = load_sparse_csr('train_tfidf_commonwords.npz')
        # symmdif = load_sparse_csr('train_tfidf_symmdiff.npz')
        train = sparse.hstack([train_sparse, bm25_sparse])
        del train_sparse, bm25_sparse
        print 'Train shape: ', train.shape

    if enable_cv:
        train, y = shuffle(train, y)
        booster.cv(train, y)
        exit()

    if use_alldata:
        print 'Using all data to fit classifier ... '
        assert num_trees > 0
        results = booster.fit_all(train, y, num_trees, features)
    else:
        print 'Using train/dev split to fit classifier ... '
        X_train, X_eval, y_train, y_eval = train_test_split(train, y,
                                                            stratify=y,
                                                            test_size=0.20,
                                                            random_state=seed)

        if oversampling > 0:
            print 'Oversampling X_train, X_eval datasets ... '
            X_train, y_train = oversample_sparse(X_train, y_train, p=oversampling)
            X_eval, y_eval = oversample_sparse(X_eval, y_eval, p=oversampling)

        results = booster.fit(X_train, X_eval, y_train, y_eval, features)
        y_pred = booster.predict(X_eval)
        print log_loss(y_eval, y_pred)
        print y_pred

    train = None
    y = None
    del train
    del y

    return results


def make_submission_name(base_name):
    return base_name + str(time.time()) + '.csv'


def boost():
    constant_seed = np.random.randint(0, 999)
    print 'Seed:', constant_seed
    use_tfidf = True
    enable_cv = False
    show_importance = not use_tfidf
    use_all_data = False
    num_trees = 1604
    oversample_p = -1.0  # 0.1546
    do_calibrate = oversample_p < 0.0
    train_by_folds = True

    # Train
    booster = LightGBMBooster(params_light, constant_seed, show_importance=show_importance)
    # booster = XGBooster(params_xgb, constant_seed, show_importance=show_importance)
    _ = train_boost(booster, seed=constant_seed, oversampling=oversample_p,
                    use_tfidf=use_tfidf, enable_cv=enable_cv, use_alldata=use_all_data, num_trees=num_trees)
    exit()
    test = prepare_test()

    if use_tfidf:
        print 'Making sparse test matrix ... '
        test_sparse = sparse.csr_matrix(test.values)
        del test
        bm25_test = load_sparse_csr('bm25_test.npz')
        test = sparse.hstack([test_sparse, bm25_test])
        del test_sparse, bm25_test

        print 'ok'
    print test.shape
    y_pred = booster.predict(test)

    # Make submission
    modelprefix = 'XGB' if isinstance(booster, XGBooster) else 'LGB'
    tfidfprefix = 'tfidf' if use_tfidf == True else ''
    alldatapref = 'alldata' if use_all_data == True else ''
    oversamplepref = 'oversampled' if oversample_p > 0 else ''
    submission_name = 'sumbit_{0}_{1}_{2}_{3}_{4}-.csv'.format(oversamplepref, alldatapref, modelprefix,
                                                              tfidfprefix, TODAY)
    make_submission(submission_name, y_pred)
    if do_calibrate:
        print 'Calibrating ... '
        calibrate(submission_name, modifier_stat_efficient)


def main():
    boost()

    # w1 = 0.5
    # w2 = 1.0 - w1
    # outname = 'mix{0}+{1}_our_best.csv'.format(int(w1 * 10), int(w2 * 10), TODAY)
    # mix(fname1='6_our_best.csv',
    #     fname2='wa_xgb_5bag_166f_20170519.csv', #best_0.1531.csv
    #     w1=w1,
    #     w2=w2,
    #     output_name=outname)
    # calibrate(outname, modifier_magic)


if __name__ == '__main__':
    main()
