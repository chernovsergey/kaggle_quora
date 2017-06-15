from scipy.stats import pearsonr

import pandas as pd
import numpy as np
import re
import string

from scipy.sparse import csr_matrix

INPUT_FOLDER = '../../data'

path = '../input/'


def remove_punct(val):
    # remove all punctuation chars
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = regex.sub('', val).lower()

    return sentence


def clean_dataframe(data):
    # first remove punctuation than make lowercase
    for col in ['question1', 'question2']:
        data[col] = data[col].apply(remove_punct)

    return data


def correlation_matrix(df, labels):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Quora Feature Correlation')
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_yticklabels(labels, fontsize=10)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.7, .75, .8, .85, .90, .95, 1])
    plt.show()


def feature_correlation(df_train):
    corr = df_train.corr()
    correlation_matrix(corr, corr.columns)
    corr.to_csv('feature_correlations.csv')


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def tfidf():
    pass
    # print 'Counting tf-idf ... '
    # train1 = pd.read_csv(INPUT_FOLDER + "/train_question1_stemmed.csv")
    # train2 = pd.read_csv(INPUT_FOLDER + "/train_question2_stemmed.csv")
    # train_main = pd.DataFrame()
    # train_main['question1'] = pd.Series(train1.values[:].ravel())
    # train_main['question2'] = pd.Series(train2.values[:].ravel())
    #
    # test1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
    # test2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
    # test_main = pd.DataFrame()
    # test_main['question1'] = pd.Series(test1.values[:].ravel())
    # test_main['question2'] = pd.Series(test2.values[:].ravel())
    #
    # tfidf = TfidfVectorizer(stop_words='english')
    # corpus = train_main.apply(lambda x: str(x['question1']) + ' ' + str(x['question2']), axis=1).values.tolist()
    # corpus_test = test_main.apply(lambda x: str(x['question1']) + ' ' + str(x['question2']), axis=1).values.tolist()
    # tfidf.fit(corpus + corpus_test)
    #
    # tfidf_train = tfidf.transform(corpus)
    # tfidf_test = tfidf.transform(corpus_test)
    #
    # print tfidf_train.shape
    # print tfidf_test.shape
    #
    # save_sparse_csr('tfidf_stem_train', tfidf_train)
    # save_sparse_csr('tfidf_stem_test', tfidf_test)
    # exit()


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def feature_filter(features):
    features = filter(lambda x: 'id' != x, features)
    features = filter(lambda x: 'is_duplicate' != x, features)
    print "\n".join(sorted(set(features)))
    print len(features), ' features total'
    return features


def prepare_train():
    print 'Loading train dataset ... '

    y = pd.read_csv(INPUT_FOLDER + '/train.csv', usecols=['is_duplicate']).values.ravel()

    df_train = pd.read_csv('train_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/trainftr.csv', usecols=['train'])
    df_train['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_train.csv', index_col=0)
    df_train = df_train.join(mephistoph)

    siamese_stack = pd.read_csv(INPUT_FOLDER + '/features/siamese_oof_train.csv', index_col=0)
    df_train = df_train.join(siamese_stack)

    train_leak = pd.read_csv(INPUT_FOLDER + '/features/drive-download/train_leak.csv', index_col=0)
    df_train = df_train.join(train_leak)

    bm25 = pd.read_csv(INPUT_FOLDER + '/features/drive-download/data_bmp25_oof.csv')
    df_train = df_train.join(bm25)

    cities = pd.read_csv(INPUT_FOLDER + '/features/drive-download/data_cities.csv')
    df_train = df_train.join(cities)

    countries = pd.read_csv(INPUT_FOLDER + '/features/drive-download/data_countries.csv')
    df_train = df_train.join(countries)

    poss_diff = pd.read_csv(INPUT_FOLDER + '/features/drive-download/data_pos_diff.csv')
    df_train = df_train.join(poss_diff)

    tifidfoof = pd.read_csv(INPUT_FOLDER + '/features/drive-download/data_tfidf_oof.csv')
    df_train = df_train.join(tifidfoof)

    features = feature_filter(df_train.columns)
    to_drop = set(df_train.columns).difference(features)
    df_train.drop(to_drop, axis=1, inplace=True)
    print 'train shape: ', df_train.shape
    # assert len(features) == df_train.shape[1]
    # df_train.to_csv('train_features_1205.csv')
    print 'ok'
    return df_train, y, features


def prepare_test():
    print 'Loading test set ...'

    df_test = pd.read_csv('test_features_1205.csv', index_col=0)

    ftr = pd.read_csv(INPUT_FOLDER + '/features/testftr.csv', usecols=['test'])
    df_test['ftr'] = ftr

    mephistoph = pd.read_csv(INPUT_FOLDER + '/features/fs6_test.csv', index_col=0)
    df_test = df_test.join(mephistoph)

    oof_bm25 = np.load('boost_sparsebm25_test.csv.npy')
    df_test['oof_bm25'] = pd.Series(oof_bm25.ravel())
    
    oof_tfidf = np.load('boost_sparsetfidf_test.csv.npy')
    df_test['oof_tfidf'] = pd.Series(oof_tfidf.ravel())

    logreg_oof = np.load('logreg_test.csv.npy')
    df_test['logreg_oof'] = pd.Series(logreg_oof.ravel())

    magic1_cosine_extra = pd.read_csv(INPUT_FOLDER + '/features/magic_cosine_extra_test.csv', index_col=0)
    df_test = df_test.join(magic1_cosine_extra)

    siamese_stack = pd.read_csv(INPUT_FOLDER + '/features/siamese_oof_test.csv', index_col=0)
    df_test = df_test.join(siamese_stack)

    magicv2 = pd.read_csv('magicv2_test.csv', index_col=0)
    df_test = df_test.join(magicv2)

    oof_mlp = pd.read_csv(INPUT_FOLDER + '/features/mlp_oof_test.csv', usecols=['MLP_oof'])
    df_test = df_test.join(oof_mlp)
    oof_mlp = None

    df_test = df_test.fillna(df_test.mean())
    df_test[df_test['wmd']==np.inf] = np.ma.masked_invalid(df_test['wmd'].values[:]).mean()
    df_test[df_test['norm_wmd'] == np.inf] = np.ma.masked_invalid(df_test['norm_wmd'].values[:]).mean()

    features = feature_filter(df_test.columns)
    to_drop = set(df_test.columns).difference(features)
    df_test.drop(to_drop, axis=1, inplace=True)
    print 'test shape:', df_test.shape
    # assert len(features) == df_test.shape[1]
    # df_test.to_csv('test_features_1205.csv')
    print 'ok'
    return df_test
