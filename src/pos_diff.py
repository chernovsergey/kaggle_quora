import pandas as pd

def pos_diff_matrix(df, max_len, prefix):
    cols = df.columns
    Q1 = df[cols[0]]
    Q2 = df[cols[1]]
    matrix = []

    def text_intersect(q1, q2, max_len):
        if isinstance(q1, float) or isinstance(q2, float):
            print q1, q2
            return []

        intersection = []

        q1_words_positions = dict()
        word_position = 0
        for word in q1.split(' '):
            q1_words_positions[word] = word_position
            word_position += 1

        word_position = 0
        for word in q2.split(' '):
            if len(intersection) < max_len:
                intersection.append(q1_words_positions.get(word, max_len * 2) - word_position)
            word_position += 1

        intersection += [max_len * 3] * (max_len - len(intersection))
        return intersection

    for q1, q2 in zip(Q1, Q2):
        matrix.append(text_intersect(q1, q2, max_len))

    return pd.DataFrame(matrix, columns=[prefix + str(i + 1) for i in range(max_len)])


MAX_WORDS_IN_QUESTION = 20
p_pos_diff = ['', '_stem_nostops']

INPUT_FOLDER = '../../data'

train_main = pd.read_csv(INPUT_FOLDER + '/train.csv')
train_main = pos_diff_matrix(train_main[['question1', 'question2']], MAX_WORDS_IN_QUESTION, 'posdiff_')
print train_main.shape
train_main.to_csv(INPUT_FOLDER + '/features/pos_diff_features_train.csv')

test_main = pd.read_csv(INPUT_FOLDER + '/test.csv')
test_main = pos_diff_matrix(test_main[['question1', 'question2']], MAX_WORDS_IN_QUESTION, 'posdiff_')
print test_main.shape
test_main.to_csv(INPUT_FOLDER + '/features/pos_diff_features_test')

