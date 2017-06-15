# coding: utf-8

# In[1]:
import tqdm
import pandas as pd


# In[3]:




def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


INPUT_FOLDER = '../data'

path = './models/input/'

train1 = pd.read_csv(INPUT_FOLDER + "/train_question1_stemmed.csv")
train2 = pd.read_csv(INPUT_FOLDER + "/train_question2_stemmed.csv")
train_main = pd.DataFrame()
train_main['question1'] = pd.Series(train1.values[:].ravel())
train_main['question2'] = pd.Series(train2.values[:].ravel())

test1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
test2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
test_main = pd.DataFrame()
test_main['question1'] = pd.Series(test1.values[:].ravel())
test_main['question2'] = pd.Series(test2.values[:].ravel())

# train_main = pd.read_csv(INPUT_FOLDER + '/train.csv')
# test_main = pd.read_csv(INPUT_FOLDER + '/test.csv')

q_ids = dict()
current_id = 1

for _,row in train_main.iterrows():
    q1 = row.question1
    q2 = row.question2

    if q1 not in q_ids:
        q_ids[q1] = current_id
        current_id+=1
    if q2 not in q_ids:
        q_ids[q2] = current_id
        current_id+=1

for _,row in test_main.iterrows():
    q1 = row.question1
    q2 = row.question2

    if q1 not in q_ids:
        q_ids[q1] = current_id
        current_id+=1
    if q2 not in q_ids:
        q_ids[q2] = current_id
        current_id+=1

train_main['qid1'] = train_main.question1.apply(lambda q: q_ids[q])
train_main['qid2'] = train_main.question2.apply(lambda q: q_ids[q])
test_main['qid1'] = test_main.question1.apply(lambda q: q_ids[q])
test_main['qid2'] = test_main.question2.apply(lambda q: q_ids[q])
print 'ok'


# In[4]:

train_main.to_csv('train_main_renumerated.csv')
test_main.to_csv('test_main_renumerated.csv')


# In[7]:

print train_main.shape
print test_main.shape
print train_main.columns
print test_main.columns


# In[11]:

train_main.head(5)


# In[12]:

test_main.head(5)


# In[14]:

cooc = dict()

for idx, row in tqdm.tqdm(train_main.iterrows()):
    id1 = row.qid1
    id2 = row.qid2
    if cooc.has_key(id1):
        cooc[id1].add(id2)
    else:
        cooc[id1] = set([id2])

    if cooc.has_key(id2):
        cooc[id2].add(id1)
    else:
        cooc[id2] = set([id1])

print 'ok train'

for idx, row in tqdm.tqdm(test_main.iterrows()):
    id1 = row.qid1
    id2 = row.qid2
    if cooc.has_key(id1):
        cooc[id1].add(id2)
    else:
        cooc[id1] = set([id2])

    if cooc.has_key(id2):
        cooc[id2].add(id1)
    else:
        cooc[id2] = set([id1])

print 'ok test'


# In[16]:

import cPickle as pickle
with open('cooc_dict.pkl', 'wb') as f:
    pickle.dump(cooc, f)


# In[17]:

len(cooc)


# In[18]:

cooc_text = dict()
for k, v in tqdm.tqdm(cooc.iteritems()):
    cooc_text[k] = " ".join(map(lambda x: str(x), v))


# In[25]:

train_main['qid1_seq'] = train_main.qid1.apply(lambda q: cooc_text[q])
train_main['qid2_seq'] = train_main.qid2.apply(lambda q: cooc_text[q])
print 'ok train'

test_main['qid1_seq'] = test_main.qid1.apply(lambda q: cooc_text[q])
test_main['qid2_seq'] = test_main.qid2.apply(lambda q: cooc_text[q])
print 'ok test'


# In[27]:

train_main.tail(5)


# In[28]:

train_main.to_csv('train_main_renumerated_seq.csv')
test_main.to_csv('test_main_renumerated_seq.csv')


### FORGET EVERITHING UP TO HERE. LOAD DATA & COMPUTE MAGIC

In[4]:

train_main = pd.read_csv('train_main_renumerated_seq.csv', usecols=['qid1_seq', 'qid2_seq'])
test_main = pd.read_csv('test_main_renumerated_seq.csv', usecols=['qid1_seq', 'qid2_seq'])


# In[6]:

from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer

corpus = pd.DataFrame(train_main['qid1_seq'].tolist()
                      + train_main['qid2_seq'].tolist()
                      + test_main['qid1_seq'].tolist()
                      + test_main['qid2_seq'].tolist()
                      , columns=['full_text'])

len(corpus)

tfidf = CountVectorizer().fit(corpus.full_text)
del corpus
print 'CountVectorizer fit ... - ok'

import multiprocessing as mp

NUM_CORES = 8


def apply_parallel_sparse(DF, my_func):
    total_rows = DF.shape[0]
    split_indices = []
    for i in range(NUM_CORES):
        split_on = (int(total_rows / NUM_CORES) * i)
        split_indices.append(split_on)
    split_indices.append(total_rows)

    DF = [DF[split_indices[i]:split_indices[i + 1]] for i in range(NUM_CORES)]

    pool = mp.Pool(NUM_CORES)

    res = []
    for part_res in pool.map(my_func, DF):
        res += part_res
    pool.close()
    pool.join()
    return res

from sklearn.metrics import  matthews_corrcoef
import numpy as np

def get_metric(DF):
    result = []
    for i in range(DF.shape[0]):
        x = DF[i, :num_features]
        y = DF[i, num_features:]
        result.append(pairwise_distances(x, y, metric='cosine')[0][0])
    return result


feature_name = 'magic_cosine_extra'


from scipy.sparse import hstack



tf_q1 = tfidf.transform(train_main['qid1_seq'])
tf_q2 = tfidf.transform(train_main['qid2_seq'])
tf_q = hstack([tf_q1, tf_q2], format='csr')
num_features = tf_q1.shape[1]
train_main[feature_name] = apply_parallel_sparse(tf_q, get_metric)
train_main[feature_name].to_csv(feature_name + '_train.csv')
print feature_name, 'train - ok'



tf_q1 = tfidf.transform(test_main['qid1_seq'])
tf_q2 = tfidf.transform(test_main['qid2_seq'])
tf_q = hstack([tf_q1, tf_q2], format='csr')
num_features = tf_q1.shape[1]
test_main[feature_name] = apply_parallel_sparse(tf_q, get_metric)
del tf_q1, tf_q2, tf_q
test_main[feature_name].to_csv(feature_name + '_test.csv')
print feature_name, 'test - ok'
