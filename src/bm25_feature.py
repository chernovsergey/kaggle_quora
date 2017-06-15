from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from models.abshishek.bm25 import BM25Transformer
from models.abshishek.prepare_datasets import save_sparse_csr

INPUT_FOLDER = '../data'
train_main = pd.read_csv(INPUT_FOLDER + '/train.csv')
test_main = pd.read_csv(INPUT_FOLDER + '/test.csv')

corpus_train = train_main.question1.values.ravel().tolist() + \
               train_main.question2.values.ravel().tolist()

corpus_test = test_main.question1.values.ravel().tolist() + \
              test_main.question2.values.ravel().tolist()

corpus = corpus_test + corpus_train
corpus = map(str, corpus)

vectorizer = CountVectorizer(min_df=1)
mat = vectorizer.fit_transform(corpus)
print len(vectorizer.vocabulary_)

b25_title = BM25Transformer(use_idf=True, k1=2.0, b=0.75)
b25_title.fit(mat)


train_rows = train_main.apply(lambda x: str(x['question1']) + " " + str(x['question2']), axis=1).values.tolist()
bm25_train = b25_title.transform(vectorizer.transform(train_rows))
save_sparse_csr('bm25_train_df1.npz', bm25_train)
print bm25_train.shape

test_rows = test_main.apply(lambda x: str(x['question1']) + " " + str(x['question2']), axis=1).values.tolist()
bm25_test = b25_title.transform(vectorizer.transform(test_rows))
save_sparse_csr('bm25_test_df1.npz', bm25_test)
print bm25_test.shape
