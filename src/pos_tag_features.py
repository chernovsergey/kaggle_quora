from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import tqdm
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import *
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import cPickle as pickle

INPUT_FOLDER = '../../data'

# train1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
# train1.fillna('', inplace=True)
# train1 = train1.values.ravel().tolist()
#
# train2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
# train2.fillna('', inplace=True)
# train2 = train2.values.ravel().tolist()


# pos_q1 = []
# for q in tqdm.tqdm(train1):
#     pos_q1.append([x[1] for x in pos_tag(word_tokenize(q))])
#
# pos_q2 = []
# for q in tqdm.tqdm(train2):
#     pos_q2.append([x[1] for x in pos_tag(word_tokenize(q))])
#
#
# with open(INPUT_FOLDER + '/features/postags_test.pkl', 'w') as f:
#     pickle.dump([pos_q1, pos_q2], f)
#
# exit()

def postag_match_share(q1, q2):
    q1words = {}
    q2words = {}
    for word in q1:
        q1words[word] = 1

    for word in q2:
        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def postag_longest_common_prefix(q1, q2):
    L = min(len(list(q1)), len(list(q2)))
    i = 0
    while i != L and q1[i] == q2[i]:
        i += 1
    return i


def dice_coeff(q1, q2):
    denominator = len(set(q1)) + len(set(q2))
    if denominator == 0:
        return 0.
    return 2 * len(set(q1).intersection(set(q2))) / float(denominator)


# match_share = []
# for pq1, pq2 in tqdm.tqdm(zip(pos_q1, pos_q2)):
#     match_share.append(postag_match_share(pq1, pq2))
# df = pd.DataFrame()
# df['postag_match_share'] = pd.Series(np.array(match_share))
# df.to_csv(INPUT_FOLDER + '/features/pos_tag_features_train.csv')
# print 'ok'
#
# lcp = []
# for pq1, pq2 in tqdm.tqdm(zip(pos_q1, pos_q2)):
#     lcp.append(postag_longest_common_prefix(pq1, pq2))

# df['postag_lcp'] = pd.Series(np.array(lcp))
# df.to_csv(INPUT_FOLDER + '/features/pos_tag_features_train.csv')


postags = pickle.load(open(INPUT_FOLDER + '/features/postags_train.pkl', 'r'))
pos_q1 = postags[0]
pos_q2 = postags[1]

tag_features_train = pd.DataFrame() #pd.read_csv(INPUT_FOLDER + '/features/pos_tag_features_train.csv')

differ_postags_count = []
symmetric_difference = []
unique_postags_q1 = []
unique_postags_q2 = []
unique_postags_q1_ratio = []
unique_postags_q2_ratio = []
dice = []
lcp = []
match_share = []
for pq1, pq2 in tqdm.tqdm(zip(pos_q1, pos_q2)):
    dice.append(dice_coeff(pq1, pq2))
    differ_postags_count.append(len(set(pq1).difference(set(pq2))))
    symmetric_difference.append(len(set(pq1).symmetric_difference(set(pq2))))
    unique_postags_q1.append(len(set(pq1)))
    unique_postags_q2.append(len(set(pq2)))

    if len(pq1) == 0:
        unique_postags_q1_ratio.append(0.)
    else:
        unique_postags_q1_ratio.append(len(set(pq1))/float(len(pq1)))

    if len(pq2) == 0:
        unique_postags_q2_ratio.append(0.)
    else:
        unique_postags_q2_ratio.append(len(set(pq2))/float(len(pq2)))

    lcp.append(postag_match_share(pq1, pq2))
    match_share.append(postag_match_share(pq1, pq2))

tag_features_train['postag_dice'] = pd.Series(dice)
tag_features_train['postag_diff_len'] = pd.Series(differ_postags_count)
tag_features_train['postag_symmetric_difference'] = pd.Series(symmetric_difference)
tag_features_train['unique_postags_q1'] = pd.Series(unique_postags_q1)
tag_features_train['unique_postags_q2'] = pd.Series(unique_postags_q2)
tag_features_train['unique_postags_q1_ratio'] = pd.Series(unique_postags_q1_ratio)
tag_features_train['unique_postags_q2_ratio'] = pd.Series(unique_postags_q2_ratio)
tag_features_train['postag_lcp'] = pd.Series(lcp)
tag_features_train['postag_match_share'] = pd.Series(match_share)
tag_features_train.to_csv(INPUT_FOLDER + '/features/pos_tag_features_train.csv')
exit()

# POS-tags tfidf
# postags = pickle.load(open(INPUT_FOLDER + '/features/postags_train.pkl', 'r'))
# pos_q1 = postags[0]
# pos_q2 = postags[1]
#
# postags = pickle.load(open(INPUT_FOLDER + '/features/postags_test.pkl', 'r'))
# pos_q1_test = postags[0]
# pos_q2_test = postags[1]
#
# # Fit tf-idf on pos-tags
# tfidf = TfidfVectorizer(stop_words='english')
# corpus = map(lambda x: " ".join(x[0] + x[1]), zip(pos_q1, pos_q2))
# corpus_test = map(lambda x: " ".join(x[0] + x[1]), zip(pos_q1_test, pos_q2_test))
# tfidf.fit(corpus + corpus_test)
# print 'fit - ok'
#
# # Train
# matrix = tfidf.transform(corpus).toarray()
# print matrix.shape
# cols = ['pos_tfidf_{0}'.format(i) for i in range(matrix.shape[1])]
# tfidf_pos = pd.DataFrame(matrix, columns=cols)
# print tfidf_pos.shape
# tfidf_pos.to_csv(INPUT_FOLDER + '/features/tfidf_postags_train.csv')
# print 'write train - ok'
#
# # Test
# matrix_test = tfidf.transform(corpus_test).toarray()
# print matrix_test.shape
# cols = ['pos_tfidf_{0}'.format(i) for i in range(matrix_test.shape[1])]
# tfidf_pos_test = pd.DataFrame(matrix_test, columns=cols)
# print tfidf_pos_test.shape
# tfidf_pos_test.to_csv(INPUT_FOLDER + '/features/tfidf_postags_test.csv')
# print 'write test - ok'