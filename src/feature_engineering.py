import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize

import stop_words

STOP_WORS = stop_words.get_stop_words('en')


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = STOP_WORS
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = STOP_WORS
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


import stop_words as sw

stop_words = sw.get_stop_words('en')


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    del M
    return v / np.sqrt((v ** 2).sum())


INPUT_FNAME = '../../data/test.csv'
FEATURES_FNAME = '../../data/quora_test_features.csv'
WV_FNAME = '../../data/GoogleNews-vectors-negative300.bin.gz'

data = pd.read_csv(INPUT_FNAME, sep=',')
if 'train' in INPUT_FNAME:
    data = data.drop(['id', 'qid1', 'qid2'], axis=1)
else:
    data = data.drop(['test_id'], axis=1)

# # ===================== SIMPLE FEATURES ==============================
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
print 'Length ok'

data['diff_len'] = data.len_q1 - data.len_q2
print 'Diff length ok'

data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
print 'Char length ok'

data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
print 'Word length ok'

data['common_words'] = data.apply(
    lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),
    axis=1)
print 'Common words count ok'

data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
print 'Fuzz ratios ok'

data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(
    lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(
    lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
print 'Fuzz partial ratios ok'

data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                          axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])),
                                           axis=1)
print 'Fuzz token ratios ok'

data.to_csv(FEATURES_FNAME, index=False)
# exit()

# ================= Part 2 ==============================
data = pd.read_csv(FEATURES_FNAME, sep=',')
print data.columns
print len(data)

model = gensim.models.KeyedVectors.load_word2vec_format(WV_FNAME, binary=True)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
data.to_csv(FEATURES_FNAME, index=False)


norm_model = gensim.models.KeyedVectors.load_word2vec_format(WV_FNAME, binary=True)
norm_model.init_sims(replace=True)
data['norm_wmd'] = data.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1)
data.to_csv(FEATURES_FNAME, index=False)

q1_vec = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question1.values)):
    q1_vec[i, :] = sent2vec(q)
model = None
np.save('../../data/q1_w2v_test', q1_vec)

q2_vec = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    q2_vec[i, :] = sent2vec(q)
model = None
np.save('../../data/q2_w2v_test', q2_vec)
print 'W2V features ok'

q1_vec = np.load('../../data/q1_w2v_test.npy')
q2_vec = np.load('../../data/q2_w2v_test.npy')

data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'Cosine ok'
#
data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'cityblock_distance ok'
#
data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'jaccard_distance ok'
#
data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'canberra_distance ok'
#
data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'euclidean_distance ok'

data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'minkowski_distance ok'

data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(q1_vec), np.nan_to_num(q2_vec))]
data.to_csv(FEATURES_FNAME, index=False)
print 'braycurtis_distance ok'

data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(q1_vec)]
data.to_csv(FEATURES_FNAME, index=False)
print 'braycurtis_distance ok'

data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(q2_vec)]
data.to_csv(FEATURES_FNAME, index=False)
print 'braycurtis_distance ok'

data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(q1_vec)]
data.to_csv(FEATURES_FNAME, index=False)
print 'braycurtis_distance ok'

data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(q2_vec)]
data.to_csv(FEATURES_FNAME, index=False)
print 'braycurtis_distance ok'

print 'All other strange fearures ok'
print len(data)
