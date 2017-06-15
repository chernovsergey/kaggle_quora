import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import MinMaxScaler

seed = 1024
np.random.seed(seed)
path = "../input/"
train = pd.read_csv(path + "train_porter.csv")

# # tfidf
train_question1_tfidf = pd.read_pickle(path + 'train_question1_tfidf.pkl')[:]
test_question1_tfidf = pd.read_pickle(path + 'test_question1_tfidf.pkl')[:]

train_question2_tfidf = pd.read_pickle(path + 'train_question2_tfidf.pkl')[:]
test_question2_tfidf = pd.read_pickle(path + 'test_question2_tfidf.pkl')[:]

train_question1_porter_tfidf = pd.read_pickle(path + 'train_question1_porter_tfidf.pkl')[:]
test_question1_porter_tfidf = pd.read_pickle(path + 'test_question1_porter_tfidf.pkl')[:]

train_question2_porter_tfidf = pd.read_pickle(path + 'train_question2_porter_tfidf.pkl')[:]
test_question2_porter_tfidf = pd.read_pickle(path + 'test_question2_porter_tfidf.pkl')[:]

train_interaction = pd.read_pickle(path + 'train_interaction.pkl')[:].reshape(-1, 1)
test_interaction = pd.read_pickle(path + 'test_interaction.pkl')[:].reshape(-1, 1)

train_porter_interaction = pd.read_pickle(path + 'train_porter_interaction.pkl')[:].reshape(-1, 1)
test_porter_interaction = pd.read_pickle(path + 'test_porter_interaction.pkl')[:].reshape(-1, 1)

train_jaccard = pd.read_pickle(path + 'train_jaccard.pkl')[:].reshape(-1, 1)
test_jaccard = pd.read_pickle(path + 'test_jaccard.pkl')[:].reshape(-1, 1)

train_porter_jaccard = pd.read_pickle(path + 'train_porter_jaccard.pkl')[:].reshape(-1, 1)
test_porter_jaccard = pd.read_pickle(path + 'test_porter_jaccard.pkl')[:].reshape(-1, 1)

train_len = pd.read_pickle(path + "train_len.pkl")
test_len = pd.read_pickle(path + "test_len.pkl")
scaler = MinMaxScaler()
scaler.fit(np.vstack([train_len, test_len]))
train_len = scaler.transform(train_len)
# test_len = scaler.transform(test_len)

print 'Loaded build in features'

w2v_cols_to_use = ["cosine_distance",
                   "cityblock_distance",
                   "jaccard_distance",
                   "canberra_distance",
                   "euclidean_distance",
                   "minkowski_distance",
                   "braycurtis_distance",
                   "skew_q1vec",
                   "skew_q2vec",
                   "kur_q1vec",
                   "kur_q2vec"]

INPUT_FOLDER = '../../data'
INPUT_TRAIN_FEATURES = INPUT_FOLDER + '/features/all_features_train_24april.csv'
df_train = pd.read_csv(INPUT_TRAIN_FEATURES)

# w2v_features = pd.read_csv(INPUT_FOLDER + '/features/train/quora_train_features.csv', usecols=w2v_cols_to_use)
# df_train = df_train.join(w2v_features)

# train_diffl = pd.read_csv(INPUT_FOLDER + '/features/diffl_sor_jac_train.csv', index_col=0)
# df_train = df_train.join(train_diffl)

#
nltk = pd.read_csv(INPUT_FOLDER + '/features/nltk_similarity_score_train.csv', index_col=0)
df_train['nltk_similarity'] = nltk['nltk_similarity_score']

sentiments = pd.read_csv(INPUT_FOLDER + '/features/sentiment_diffs.csv')
df_train['sentiment_diffs'] = sentiments['sentiment_diffs']

wordnet = pd.read_csv(INPUT_FOLDER + '/features/wordned_similarity_train.csv', index_col=0)
df_train = df_train.join(wordnet)

ncd = pd.read_csv(INPUT_FOLDER + '/features/ncd_distance_train.csv', index_col=0)
df_train = df_train.join(ncd)

magic = pd.read_csv(INPUT_FOLDER + '/features/magic_features_train.csv', usecols=['q1_freq', 'q2_freq'])
df_train = df_train.join(magic)
magic = None

devil = pd.read_csv(INPUT_FOLDER + '/features/devil_features_train.csv', usecols=['devil_cosine'])
df_train = df_train.join(devil)

magic1 = pd.read_csv(INPUT_FOLDER + '/features/magic1_train.csv', usecols=['magic1'])
df_train = df_train.join(magic1)

magic1_stem = pd.read_csv(INPUT_FOLDER + '/features/magic1_train_stem.csv', usecols=['magic1_stem'])
df_train = df_train.join(magic1_stem)
#
df_train.fillna(0, inplace=True)


INPUT_TEST_FEATURES = '../../data/features/all_features_test_24april.csv'
df_test = pd.read_csv(INPUT_TEST_FEATURES)

# test_diffl = pd.read_csv(INPUT_FOLDER + '/features/diffl_sor_jac_test.csv', index_col=0)
# df_test = df_test.join(test_diffl)

# w2v_features = pd.read_csv(INPUT_FOLDER + '/features/test/quora_test_features.csv', usecols=w2v_cols_to_use)
# df_test = df_test.join(w2v_features)


nltk = pd.read_csv(INPUT_FOLDER + '/features/nltk_similarity_score_test.csv', index_col=0)
df_test['nltk_similarity'] = nltk['nltk_similarity_score']
nltk = None

sentiments = pd.read_csv(INPUT_FOLDER + '/features/sentiment_diffs_test.csv')
df_test['sentiment_diffs'] = sentiments['sentiment_diffs']
sentiments = None

wordnet = pd.read_csv(INPUT_FOLDER + '/features/wordned_similarity_test.csv', index_col=0)
df_test = df_test.join(wordnet)
wordnet = None

ncd = pd.read_csv(INPUT_FOLDER + '/features/ncd_distance_test.csv', index_col=0)
df_test = df_test.join(ncd)
ncd = None

magic = pd.read_csv(INPUT_FOLDER + '/features/magic_features_test.csv', usecols=['q1_freq', 'q2_freq'])
df_test = df_test.join(magic)
magic = None

devil = pd.read_csv(INPUT_FOLDER + '/features/devil_features_test.csv', usecols=['devil_cosine'])
df_test = df_test.join(devil)
devil = None

magic1 = pd.read_csv(INPUT_FOLDER + '/features/magic1_test.csv', usecols=['magic1'])
df_test = df_test.join(magic1)
magic1 = None

magic1_stem = pd.read_csv(INPUT_FOLDER + '/features/magic1_test_stem.csv', usecols=['magic1_stem'])
df_test = df_test.join(magic1_stem)

df_test.fillna(0, inplace=True)

print 'Processing nan-s, inf-s'
print '...train df'
df_train = df_train.replace([np.inf, -np.inf], 0)
# assert sum(np.isnan(df_train).any(1)) == 0
# assert sum(np.isinf(df_train).any(1)) == 0

print '...test df'
df_test = df_test.replace([np.inf, -np.inf], 0)
# assert sum(np.isnan(df_test).any(1)) == 0
# assert sum(np.isinf(df_test).any(1)) == 0

print '... ok'

print 'Loaded train/test sets'

X = ssp.hstack([

    # df_train['norm_wmd'].values[:].reshape(-1, 1),
    # df_train['wmd'].values[:].reshape(-1, 1),
    # df_train['sor'].values[:].reshape(-1, 1),
    # df_train['jac'].values[:].reshape(-1, 1),
    # df_train['diffl'].values[:].reshape(-1, 1),
    # df_train['cosine_distance'].values[:].reshape(-1, 1),
    # df_train['cityblock_distance'].values[:].reshape(-1, 1),
    # df_train['jaccard_distance'].values[:].reshape(-1, 1),
    # df_train['canberra_distance'].values[:].reshape(-1, 1),
    # df_train['euclidean_distance'].values[:].reshape(-1, 1),
    # df_train['minkowski_distance'].values[:].reshape(-1, 1),
    # df_train['braycurtis_distance'].values[:].reshape(-1, 1),
    # df_train['skew_q1vec'].values[:].reshape(-1, 1),
    # df_train['skew_q2vec'].values[:].reshape(-1, 1),
    # df_train['kur_q1vec'].values[:].reshape(-1, 1),
    # df_train['kur_q2vec'].values[:].reshape(-1, 1),
    # df_train['lsh_jaccard'].values[:].reshape(-1, 1),
    # df_train['q1_freq'].values[:].reshape(-1, 1),
    # df_train['q2_freq'].values[:].reshape(-1, 1),
    df_train['devil_cosine'].values[:].reshape(-1, 1),
    df_train['magic1'].values[:].reshape(-1, 1),
    df_train['magic1_stem'].values[:].reshape(-1, 1),
    # df_train['art_cosines'].values[:].reshape(-1, 1),
    # df_train['nltk_similarity'].values[:].reshape(-1, 1),
    # df_train['sentiment_diffs'].values[:].reshape(-1, 1),
    # df_train['wordnet_similarity'].values[:].reshape(-1, 1),
    # df_train['art_cosines'].values[:].reshape(-1, 1),
    # df_train['ncd'].values[:].reshape(-1, 1),
    # df_train['ncd_stem'].values[:].reshape(-1, 1),

    train_question1_tfidf,
    train_question2_tfidf,
    train_interaction,
    train_porter_interaction,
    train_jaccard,
    train_porter_jaccard,
    train_len,
]).tocsr()

print 'X is ready'


y = train['is_duplicate'].values[:]

X_t = ssp.hstack([

    # df_test['norm_wmd'].values[:].reshape(-1, 1),
    # df_test['wmd'].values[:].reshape(-1, 1),
    # df_test['sor'].values[:].reshape(-1, 1),
    # df_test['jac'].values[:].reshape(-1, 1),
    # df_test['diffl'].values[:].reshape(-1, 1),
    # df_test['cosine_distance'].values[:].reshape(-1, 1),
    # df_test['cityblock_distance'].values[:].reshape(-1, 1),
    # df_test['jaccard_distance'].values[:].reshape(-1, 1),
    # df_test['canberra_distance'].values[:].reshape(-1, 1),
    # df_test['euclidean_distance'].values[:].reshape(-1, 1),
    # df_test['minkowski_distance'].values[:].reshape(-1, 1),
    # df_test['braycurtis_distance'].values[:].reshape(-1, 1),
    # df_test['skew_q1vec'].values[:].reshape(-1, 1),
    # df_test['skew_q2vec'].values[:].reshape(-1, 1),
    # df_test['kur_q1vec'].values[:].reshape(-1, 1),
    # df_test['kur_q2vec'].values[:].reshape(-1, 1),
    # df_test['lsh_jaccard'].values[:].reshape(-1, 1),
    # df_test['q1_freq'].values[:].reshape(-1, 1),
    # df_test['q2_freq'].values[:].reshape(-1, 1),
    df_test['devil_cosine'].values[:].reshape(-1, 1),
    df_test['magic1'].values[:].reshape(-1, 1),
    df_test['magic1_stem'].values[:].reshape(-1, 1),
    # df_test['art_cosines'].values[:].reshape(-1, 1),
    # df_test['nltk_similarity'].values[:].reshape(-1, 1),
    # df_test['sentiment_diffs'].values[:].reshape(-1, 1),
    # df_test['wordnet_similarity'].values[:].reshape(-1, 1),
    # df_test['art_cosines'].values[:].reshape(-1, 1),
    # df_test['ncd'].values[:].reshape(-1, 1),
    # df_test['ncd_stem'].values[:].reshape(-1, 1),

    test_question1_tfidf,
    test_question2_tfidf,
    test_interaction,
    test_porter_interaction,
    test_jaccard,
    test_porter_jaccard,
    test_len,
]).tocsr()

print 'X_t is ready'

print X.shape
print X_t.shape

skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

dump_svmlight_file(X, y, path + "X_tfidf.svm")
del X
dump_svmlight_file(X_t, np.zeros(X_t.shape[0]), path + "X_t_tfidf.svm")
del X_t


def oversample(X_ot, y, p=0.165):
    pos_ot = X_ot[y == 1]
    neg_ot = X_ot[y == 0]
    # p = 0.165
    scale = ((pos_ot.shape[0] * 1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -= 1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y = np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]] = 1.0
    print y.mean()
    return ot, y


X_train, y_train = oversample(X_train.tocsr(), y_train, p=0.165)
X_test, y_test = oversample(X_test.tocsr(), y_test, p=0.165)

X_train, y_train = shuffle(X_train, y_train, random_state=seed)

dump_svmlight_file(X_train, y_train, path + "X_train_tfidf.svm")
dump_svmlight_file(X_test, y_test, path + "X_test_tfidf.svm")
