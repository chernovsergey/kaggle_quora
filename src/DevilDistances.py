
# coding: utf-8

# In[1]:

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

import tqdm
import pandas as pd
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

INPUT_FOLDER = '../data'

train1 = pd.read_csv(INPUT_FOLDER + "/train_question1_stemmed.csv")
train2 = pd.read_csv(INPUT_FOLDER + "/train_question2_stemmed.csv")
train_stemm = pd.DataFrame()
train_stemm['question1'] = pd.Series(train1.values[:].ravel())
train_stemm['question2'] = pd.Series(train2.values[:].ravel())

test1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
test2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
test_stemm = pd.DataFrame()
test_stemm['question1'] = pd.Series(test1.values[:].ravel())
test_stemm['question2'] = pd.Series(test2.values[:].ravel())


# In[2]:

H = load_sparse_csr('devil_matrix.npz')


# In[3]:

unique_q = set(train1.values.ravel().tolist() + test1.values.ravel().tolist() + train2.values.ravel().tolist() + test2.values.ravel().tolist())
print len(unique_q)
unique_q_id = dict()
for i, q in tqdm.tqdm(enumerate(unique_q)):
    unique_q_id[q] = i


# In[4]:

from sklearn.preprocessing import normalize
def cosine_sparse(a, b):
    a = normalize(a, norm='l2', axis=1)
    b = normalize(b, norm='l2', axis=1)
    return np.sum((a.dot(b.T)))


# In[5]:

from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis, dice, hamming, correlation

devil_euclid = []
devil_manhat = []
devil_cosine = []
for idx, row in tqdm.tqdm(train_stemm.iterrows()):
    id_1 = unique_q_id[row['question1']]
    id_2 = unique_q_id[row['question2']]
    a_vec = H[id_1]
    b_vec = H[id_2]
    devil_euclid.append(np.sqrt(np.sum((a_vec - b_vec).data**2)))
    devil_manhat.append(np.sum(abs((a_vec - b_vec).data)))
    devil_cosine.append(cosine_sparse(a_vec, b_vec))


# In[ ]:

devil_df = pd.DataFrame()
devil_df['devil_euclid'] = pd.Series(devil_euclid)
devil_df['devil_manhat'] = pd.Series(devil_manhat)
devil_df['devil_cosine'] = pd.Series(devil_cosine)
devil_df.to_csv('devil_features_train.csv')


# In[ ]:

devil_euclid_test = []
devil_manhat_test = []
devil_cosine_test = []
for idx, row in tqdm.tqdm(test_stemm.iterrows()):
    id_1 = unique_q_id[row['question1']]
    id_2 = unique_q_id[row['question2']]
    a_vec = H[id_1]
    b_vec = H[id_2]
    devil_euclid_test.append(np.sqrt(np.sum((a_vec - b_vec).data**2)))
    devil_manhat_test.append(np.sum(abs((a_vec - b_vec).data)))
    devil_cosine_test.append(cosine_sparse(a_vec, b_vec))

# In[ ]:

assert len(devil_euclid_test) == test_stemm.shape[0]
devil_df = pd.DataFrame()
devil_df['devil_euclid'] = pd.Series(devil_euclid)
devil_df['devil_manhat'] = pd.Series(devil_manhat)
devil_df['devil_cosine'] = pd.Series(devil_cosine)
devil_df.to_csv('devil_features_test.csv')

