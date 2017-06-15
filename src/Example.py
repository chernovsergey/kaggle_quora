
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import datetime
import os
import pickle

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic(u'pylab inline')
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from scipy.sparse import csr_matrix, coo_matrix,hstack, vstack
DATA_PATH = 'data/final/'


# In[2]:

questenable_features = ['diff_avg_word', 'porter_jaccard', 'lsh_jaccard'
                        ,'art_cosines', 'nltk_similarity', 'wordnet_similarity', 'ncd_stem', 'ncd']
sergey_features = ['magic1', 'devil_cosine','magic1_stem', 'len_fm_0', 'len_fm_1', 'len_fm_2', 'len_fm_3', 'len_fm_4'
                  ,'len_fm_5', 'len_fm_6', 'len_fm_7', 'len_fm_8'] + questenable_features

wordnet_features = ['max_similarity', 'mean_similarity', 'max_lch_similarity', 'mean_lch_similarity', 
                    'max_res_similarity', 'mean_res_similarity']
pos_tags_features = ['pos_tag_cosine_dist', 'pos_tag_manhattan_dist', 'pos_tag_jaccard_dist', 'pos_tag_canberra_dist',
                     'pos_tag_euclidean_dist', 'pos_tag_minkowski_dist', 'pos_tag_braycurtis_dist', 
                     'pos_tag_chebyshev_dist', 'pos_tag_correlation_dist', 'pos_tag_hamming_dist', 
                     'pos_tag_skewness_q1', 'pos_tag_skewness_q2', 'pos_tag_kurtosis_q1', 'pos_tag_kurtosis_q2']
abisheck_features = ['cosine_distance', 'cityblock_distance', 'jaccard_distance', 'canberra_distance', 
                     'euclidean_distance', 'minkowski_distance', 'braycurtis_distance', 'skew_q1vec', 'skew_q2vec', 
                     'kur_q1vec', 'kur_q2vec'] # not in dataset
location_features = ['z_city_match', 'z_city_match_num', 'z_city_mismatch', 'z_city_mismatch_num', 'z_q1_city_num', 
                     'z_q1_has_city', 'z_q2_city_num', 'z_q2_has_city', 'z_country_match', 'z_country_match_num', 
                     'z_country_mismatch', 'z_country_mismatch_num', 'z_q1_country_num', 'z_q1_has_country', 
                     'z_q2_country_num', 'z_q2_has_country']
oof_features = ['/data/pl0_L_XGB_001_F5_B1_34952', '/data/pl0_XGB_001_F5_B1_31328', '/data/pl0_XGB_004_F5_B1_38577', 
                '/data/pl0_XGB_004_F5_B1_38641', '/data/pl0_XGB_005_F5_B1_41729', '/data/pl0_fb_XGB_004_F5_B1_38241', 
                'bmp25_all_oof', 'bmp25_oof', 'bmp25_pca_oof', 'ftr_nn', 'm_diff_q1_q2_tf_oof', 
                'm_q1_q2_tf_svd100_oof', 'm_vstack_svd_absdiff_q1_q2_oof', 'm_vstack_svd_mult_q1_q2_oof', 
                'm_w1l_tfidf_oof', 'tfidf_all_oof', 'tfidf_oof', 'tfidf_pca_oof']
good_magic = ['dist_jaccard', 'ids_inter_share_lev2', 'q1_cluster', 'q1_hash_neighbour2_n','q1_triangle','q2_triangle',
              'q2_cluster', 'q2_hash_neighbour2_n', 'qmax_ids_unique_share_lev2', 'qmin_ids_counts_med_lev2',
              'dist_pref_attach', 'dist_rai', 'ids_inter_share', 'q1_count_in_q1', 'q1_di_eig_cent',
              'q1_freq_norm', 'q1_hash_neighbour2_max', 'q1_hash_neighbour2_std', 'q1_hash_neighbour_n',
              'q2_hash_neighbour_n', 'q1_hash_neighbour_std', 'q1_ids_counts_avg_lev2', 'q1_sq_cluster',
              'q2_sq_cluster','q2_count_in_q1','q2_count_in_q2','q2_di_eig_cent','q2_freq_norm',
              'q2_hash_neighbour2_max','q2_hash_neighbour_norm','q2_in_q1_share','qmax_ids_counts_avg',
              'qmax_ids_counts_min', 'qmin_count_all', 'qmin_ids_counts_avg', 'qmin_ids_counts_max_lev2', 
              'qmin_ids_counts_med']
bad_magic = ['magic1_stem', 'q1_count_in_q1', 'q1_eig_cent', 'q2_eig_cent', 
             'qmax_ids_counts_med_lev2.1', 'qmin_ids_counts_med_lev2.1']

print (len(good_magic))


# In[3]:

LOAD_TEST = False

ytrain = pd.read_csv(os.path.join(DATA_PATH,'target.csv')).target

train = pd.concat((pd.read_csv(os.path.join(DATA_PATH, 'train_nlp.csv')),
                   #pd.read_csv(os.path.join(DATA_PATH, 'train_magic.csv'), usecols=good_magic),
                   #pd.read_csv(os.path.join(DATA_PATH, 'train_oof.csv')),
                   #pd.read_csv(os.path.join(DATA_PATH, 'other', 'data_abisheck.csv'), sep = ',', encoding = 'cp1251', usecols=abisheck_features),
                  ),axis=1, copy=False)
print (train.shape, train.dropna().shape, ytrain.shape)
#print ([f for f in list(train.columns) if 'Unnamed' in f])
train = train.loc[:,~train.columns.duplicated()]
print (train.shape, train.dropna().shape, ytrain.shape)
train.fillna(-1, inplace=True)
train.replace([-np.inf, np.inf], -1, inplace=True)
num_features = [f for f in list(train.columns) if 'Unnamed' not in f]
print (len(num_features))

if LOAD_TEST:
    test = pd.concat(( pd.read_csv(os.path.join(DATA_PATH, 'test_nlp.csv')),
                       pd.read_csv(os.path.join(DATA_PATH, 'test_magic.csv'), usecols=good_magic),
                       pd.read_csv(os.path.join(DATA_PATH, 'test_oof.csv')),
                       #pd.read_csv(os.path.join(DATA_PATH, 'other', 'kagg_abisheck.csv'), sep = ',', encoding = 'cp1251', usecols=abisheck_features),
                  ),axis=1, copy=False)
    test = test .loc[:,~test .columns.duplicated()]
    test.fillna(-1,inplace=True)
    test.replace([-np.inf, np.inf], -1, inplace=True)
    print (test.shape)


# In[4]:

## Делаем две холдаут выборки если тест не грузили. Получаем train, valid, test
if not LOAD_TEST:    
    ids = pd.read_csv('data/final/data_ids.csv',usecols=['graph_id'])
    graph_ids_unique = ids.graph_id.unique()
    kf = KFold(n_splits=5, shuffle=True)
    train_graphs, test_graphs = list(kf.split(graph_ids_unique))[0]
    train_ind = ids[ids.graph_id.isin(graph_ids_unique[train_graphs])].index.values
    test_ind  = ids[ids.graph_id.isin(graph_ids_unique[test_graphs ])].index.values
    
    kf_valid = KFold(n_splits=2, shuffle=True)
    graph_ids_test = graph_ids_unique[test_graphs]
    test_graphs, valid_graphs = list(kf_valid.split(graph_ids_test))[0]
    test_ind  = ids[ids.graph_id.isin(graph_ids_test[test_graphs] )].index.values
    valid_ind = ids[ids.graph_id.isin(graph_ids_test[valid_graphs])].index.values

    test  = train.iloc[test_ind]
    valid = train.iloc[valid_ind]
    train = train.iloc[train_ind]
    
    ytest  = ytrain[test_ind]
    yvalid = ytrain[valid_ind]
    ytrain = ytrain[train_ind]
    
    print (train.shape, test.shape, valid.shape)


# ### Запускаем модель.
# 
# 1) Обучаемся на трейне, количество итераций выбираем по valid, предсказываем test
# 
# 2) Обучаемся на трейне, количество итераций выбираем по test, предсказываем valid
# 
# 3) Возвращаем предсказания, модель и средний лосс. Средний лосс считаем финальным качеством дл этого набора фичей
# 
# 4) одно такое предсказание занимает 2-3 минуты

# In[6]:

import lightgbm as lgb
def train_with_valid(num_features):
    dtrain = lgb.Dataset(train[num_features], ytrain, max_bin=350)
    dvalid = lgb.Dataset(valid[num_features], yvalid, reference=dtrain)
    dtest  = lgb.Dataset(test [num_features], ytest , reference=dtrain)
    
    params = {
        'task': 'train','boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'binary_logloss',
        'feature_fraction': 1,#0.95,
        'min_data_in_leaf': 10, 
        'bagging_freq': 3, 
        'min_gain_to_split': 0, 
        'lambda_l2': 1, 
        'learning_rate': 0.1, 
        'num_leaves': 128, 
        'bagging_fraction': 1}#0.85}
    
    gbm = lgb.train(params, dtrain, num_boost_round=10000, early_stopping_rounds=10,
                    valid_sets=[dtrain, dvalid], verbose_eval=False)
    test_pred = gbm.predict(test[num_features], num_iteration=gbm.best_iteration)
    
    gbm = lgb.train(params, dtrain, num_boost_round=10000, early_stopping_rounds=10,
                    valid_sets=[dtrain, dtest], verbose_eval=False)
    valid_pred = gbm.predict(valid[num_features], num_iteration=gbm.best_iteration)
    
    test_score  = log_loss(ytest,  test_pred)
    valid_score = log_loss(yvalid, valid_pred)
    #print ('scores:', round(test_score,5),  round(valid_score,5))
    #print (round((test_score+valid_score)/2, 5))
    return test_pred, valid_pred, gbm, (test_score+valid_score)/2

#test_pred, valid_pred, gbm, score = train_with_valid(num_features)
#print (score)


# In[ ]:

bad_features = []
really_good_features = []
suggested_features = set(num_features) - set(bad_features)
test_pred, valid_pred, gbm, base_score = train_with_valid(list(suggested_features))
print ('base_score:', base_score)

while True:
    imprvs = dict()
    #Набор фичей базового уровня (сет)
    suggested_features = set(num_features) - set(bad_features)
    
    # Кандидаты на удаление (сет)
    candidates_for_removal = suggested_features - set(really_good_features) 
    
    # итерируемся по всем фичам, но в случайном порядке
    for f in np.random.choice(list(candidates_for_removal), size=len(candidates_for_removal), replace=False):
        #if 'posdiff_' not in f: # можно убрать условие. Просто этих фичей 40 штук и вроде все дают
        if True:
            test_pred, valid_pred, gbm, score = train_with_valid(list(set(suggested_features) - set([f])))
            improvement = (base_score - score)
            imprvs[f] = improvement
            
            # если прирост мощный, то сразу добавляем фичу в плохие, обновляем suggested_features и base_score
            if improvement>0.0001:
                bad_features.append(f)
                suggested_features = set(num_features) - set(bad_features)
                base_score = score
                print (round(improvement,6), f)
                print (bad_features)
                np.save(os.path.join(DATA_PATH, 'bad_features.npy'),np.array(list(bad_features)))

            # если падение очень мощное, то добавляем фичу в really_good_features 
            # давая таким ей иммунитет на удаление на следующих итерациях
            if improvement<-0.001:
                really_good_features.append(f)
                print ('good feature:', round(improvement,6), f)
                np.save(os.path.join(DATA_PATH, 'really_good_features.npy'),np.array(list(really_good_features)))
                
            # если прирост есть - просто выводим на экран
            if improvement>0.00001:   
                print (round(improvement,6), round(score,5), f) 
            
    # добавляем в bad_features самую плохую фичу после одной проходки. 
    # При этом она уже может в них быть, если дала мощный рпирост
    for i in sorted(imprvs.items(), key=lambda x: -x[1])[:1]:
        feature = i[0]
        improvement = i[1]
        if feature not in bad_features:
            bad_features.append(feature)
            np.save(os.path.join(DATA_PATH, 'bad_features.npy'),np.array(list(bad_features)))
    # обновляем base_score
    test_pred, valid_pred, gbm, base_score = train_with_valid(list(set(suggested_features) - set(bad_features)))
    print (round(score,5), round(improvement,5), bad_features, '\n')


# In[ ]:

print (really_good_features)
print (bad_features)


# In[16]:

suggested_features2 = np.load(os.path.join(DATA_PATH, 'sugg.npy'))


# In[18]:

suggested_features2


# In[ ]:



