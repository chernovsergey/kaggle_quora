from datasketch import MinHash, WeightedMinHash
from nltk import ngrams
import pandas as pd
import itertools as it
from tqdm import tqdm

INPUT_FNAME = '../../data/test.csv'
OUTPUT_FEATURES = '../../data/features/lhs_jaccard_test.csv'

data = pd.read_csv(INPUT_FNAME, sep=',')

Q1 = data.question1.values
Q2 = data.question2.values
import multiprocessing

thread_safe_queue = multiprocessing.Queue()


def worker(pair):
    q1 = pair[0]
    q2 = pair[1]
    try:
        minhash1 = MinHash(num_perm=256)
        for d in ngrams(q1, 3):
            minhash1.update("".join(d))

        minhash2 = MinHash(num_perm=256)
        for d in ngrams(q2, 3):
            minhash2.update("".join(d))

        wmh = WeightedMinHash(42, minhash1.hashvalues)
        wmh2 = WeightedMinHash(42, minhash2.hashvalues)
        thread_safe_queue.put(wmh.jaccard(wmh2))

        if thread_safe_queue.qsize() % 10000 == 0:
            print thread_safe_queue.qsize()
    except:
        print q1, q2
        thread_safe_queue.put(0.)


sent_pairs = zip(Q1, Q2)
pool = multiprocessing.Pool(processes=8)
pool.map(worker, sent_pairs)
print thread_safe_queue.qsize()
pool.close()

print 'Scores computed'

all_scores = [thread_safe_queue.get() for _ in xrange(thread_safe_queue.qsize())]
df = pd.DataFrame()
df['lhs_jaccard'] = pd.Series(all_scores)
df['lhs_jaccard'].to_csv(OUTPUT_FEATURES)
print len(df['lhs_jaccard'])
print 'ok' if len(all_scores) == len(sent_pairs) else 'ups'
