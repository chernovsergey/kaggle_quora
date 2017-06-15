import zlib
import pandas as pd
INPUT_FOLDER = '../../data/'

def compressed_len(s):
    return len(zlib.compress(s))

def ncd(x, y):
    if isinstance(x, float):
        x = str(x)
    if isinstance(y, float):
        y = str(y)

    x = x.strip()
    y = y.strip()
    C_x = compressed_len(x)
    C_y = compressed_len(y)
    C_xy = compressed_len("".join([x,y]))
    result = (C_xy - min(C_x, C_y)) / float(max(C_x, C_y))
    return result

train1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
train2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
train_stemm = pd.DataFrame()
train_stemm['question1'] = pd.Series(train1.values[:].ravel())
train_stemm['question2'] = pd.Series(train2.values[:].ravel())

train = pd.read_csv(INPUT_FOLDER + '/test.csv')

ncd_result = pd.DataFrame()
ncd_result['ncd_stem'] = train_stemm.apply(lambda row: ncd(row['question1'], row['question2']), axis=1)
ncd_result['ncd'] = train.apply(lambda row: ncd(row['question1'], row['question2']), axis=1)
ncd_result.to_csv(INPUT_FOLDER + '/features/ncd_distance_test.csv')
