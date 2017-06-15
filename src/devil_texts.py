import pandas as pd
import numpy as np
import zlib
from datasketch import MinHash
from datasketch import WeightedMinHash
from nltk import word_tokenize, ngrams
from fuzzywuzzy import fuzz

INPUT_FNAME = 'test_main_renumerated_seq.csv'
FEATURES_FNAME = 'test_devil_texts.csv'

data = pd.read_csv(INPUT_FNAME, sep=',', index_col=0)
# data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
# data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
# print 'Length ok'
#
# data['diff_len'] = data.len_q1 - data.len_q2
# print 'Diff length ok'

# data['len_char_q1'] = data.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# data['len_char_q2'] = data.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
# print 'Char length ok'

# data['len_word_q1'] = data.question1.apply(lambda x: len(str(x).split()))
# data['len_word_q2'] = data.question2.apply(lambda x: len(str(x).split()))
# print 'Word length ok'

data['common_words'] = data.apply(
    lambda x: len(set(str(x['qid1_seq']).lower().split()).intersection(set(str(x['qid2_seq']).lower().split()))),
    axis=1)
print 'Common words count ok'

# data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['qid1_seq']), str(x['qid2_seq'])), axis=1)
# data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['qid1_seq']), str(x['qid2_seq'])), axis=1)
# print 'Fuzz ratios ok'
#
# data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['qid1_seq']), str(x['qid2_seq'])), axis=1)
# data['fuzz_partial_token_set_ratio'] = data.apply(
#     lambda x: fuzz.partial_token_set_ratio(str(x['qid1_seq']), str(x['qid2_seq'])), axis=1)
# data['fuzz_partial_token_sort_ratio'] = data.apply(
#     lambda x: fuzz.partial_token_sort_ratio(str(x['qid1_seq']), str(x['qid2_seq'])), axis=1)
# print 'Fuzz partial ratios ok'

data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['qid1_seq']), str(x['qid2_seq'])),
                                          axis=1)
# data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['qid1_seq']), str(x['qid2_seq'])),
#                                            axis=1)
print 'Fuzz token ratios ok'


# Q1 = data.qid1_seq.values
# Q2 = data.qid2_seq.values
# import multiprocessing
#
# thread_safe_queue = multiprocessing.Queue()
#
#
# def worker(pair):
#     q1 = pair[0]
#     q2 = pair[1]
#     try:
#         minhash1 = MinHash(num_perm=256)
#         for d in ngrams(q1, 3):
#             minhash1.update("".join(d))
#
#         minhash2 = MinHash(num_perm=256)
#         for d in ngrams(q2, 3):
#             minhash2.update("".join(d))
#
#         wmh = WeightedMinHash(42, minhash1.hashvalues)
#         wmh2 = WeightedMinHash(42, minhash2.hashvalues)
#         thread_safe_queue.put(wmh.jaccard(wmh2))
#
#         if thread_safe_queue.qsize() % 10000 == 0:
#             print thread_safe_queue.qsize()
#     except:
#         print q1, q2
#         thread_safe_queue.put(0.)
#
#
# sent_pairs = zip(Q1, Q2)
# pool = multiprocessing.Pool(processes=8)
# pool.map(worker, sent_pairs)
# print thread_safe_queue.qsize()
# pool.close()
#
# all_scores = [thread_safe_queue.get() for _ in xrange(thread_safe_queue.qsize())]
# data['lhs_jaccard'] = pd.Series(all_scores)
# print 'ok' if len(all_scores) == len(sent_pairs) else 'ups'
#
#
# def compressed_len(s):
#     return len(zlib.compress(s))
#
# def ncd(x, y):
#     if isinstance(x, float):
#         x = str(x)
#     if isinstance(y, float):
#         y = str(y)
#
#     x = x.strip()
#     y = y.strip()
#     C_x = compressed_len(x)
#     C_y = compressed_len(y)
#     C_xy = compressed_len("".join([x,y]))
#     result = (C_xy - min(C_x, C_y)) / float(max(C_x, C_y))
#     return result
#
# data['ncd'] = data.apply(lambda row: ncd(row['qid1_seq'], row['qid2_seq']), axis=1)
data.to_csv(FEATURES_FNAME, index=False)