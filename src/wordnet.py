from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import tqdm

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        seq = [synset.path_similarity(ss) for ss in synsets2]
        best_score = max(seq) if len(seq) != 0 else 0

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    if count == 0:
        return 0.

    # Average the values
    score /= count
    return score

import pandas as pd
INPUT_FOLDER = '../../data'

train1 = pd.read_csv(INPUT_FOLDER + "/test_question1_stemmed.csv")
train1.fillna('', inplace=True)
train1 = train1.values.ravel().tolist()

train2 = pd.read_csv(INPUT_FOLDER + "/test_question2_stemmed.csv")
train2.fillna('', inplace=True)
train2 = train2.values.ravel().tolist()

assert len(train1) == len(train2)
print 'ok'

wordnet = []
for a, b in tqdm.tqdm(zip(train1, train2)):
    wordnet.append(sentence_similarity(a,b))

df = pd.DataFrame()
df['wordnet_similarity'] = pd.Series(wordnet)
df.to_csv(INPUT_FOLDER + '/features/wordned_similarity_test.csv')