from __future__ import division

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import StratifiedKFold

INPUT_FOLDER = '../../data'
path = '../input/'


def prepare_train():
    df_train = pd.read_csv('train_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/trainftr.csv', usecols=['train'])
    df_train['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_train.csv', index_col=0)
    df_train = df_train.join(mephistoph)

    oof_bm25 = np.load('boost_sprasebm25_train.csv.npy')
    df_train['oof_bm25'] = pd.Series(oof_bm25.ravel())

    oof_tfidf = np.load('boost_sprasetfidf_train.csv.npy')
    df_train['oof_tfidf'] = pd.Series(oof_tfidf.ravel())

    return df_train


def prepare_test():
    df_test = pd.read_csv('test_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/testftr.csv', usecols=['test'])
    df_test['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_test.csv', index_col=0)
    df_test = df_test.join(mephistoph)

    oof_bm25 = np.load('boost_sparsebm25_test.csv.npy')
    df_test['oof_bm25'] = pd.Series(oof_bm25.ravel())

    oof_tfidf = np.load('boost_sparsetfidf_test.csv.npy')
    df_test['oof_tfidf'] = pd.Series(oof_tfidf.ravel())

    return df_test


if __name__ == '__main__':

    RANDOM_STATE = 542
    N_FOLDS = 5

    # Right way
    XTrain = prepare_train()
    XTrain.fillna(0, inplace=True)
    XTrain.replace([np.inf, -np.inf], 0, inplace=True)

    YTrain = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values

    XTest = prepare_test()
    XTest.fillna(0, inplace=True)
    XTest.replace([np.inf, -np.inf], 0, inplace=True)

    features = XTrain.columns
    print len(features), ' features total'
    print "\n".join(features)

    XTrain = XTrain.ix[:, features].as_matrix()
    YTrain = np.array(YTrain).ravel()

    XTest = XTest.ix[:, features].as_matrix()

    skf = StratifiedKFold(n_splits=N_FOLDS).split(XTrain, YTrain)
    splits = list(skf)

    base_classifiers = [RidgeClassifier()]

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
            y_pred = clf.predict_proba(XTrain[predict_indexes])[:, 1]
            y_pred = y_pred.ravel()
            df_blend_train[predict_indexes, clf_id] = y_pred
            lloss = log_loss(YTrain[predict_indexes], y_pred)
            oof_loglosses[clf_id, fold_id] = lloss
            print 'LogLoss: ', lloss

            # Predict on entire test set
            y_pred_test = clf.predict_proba(XTest)[:, 1]
            y_pred_test = y_pred_test.ravel()
            # y_pred_test = modifier_stat_efficient(y_pred_test)
            dataset_blend_test_j[:, fold_id] = y_pred_test

        # Average predictions for test set
        df_blend_test[:, clf_id] = dataset_blend_test_j.mean(1)

    print "Out of fold logloss-es:\n", oof_loglosses

    np.save('logreg_train.csv', df_blend_train)
    np.save('logreg_test.csv', df_blend_test)
