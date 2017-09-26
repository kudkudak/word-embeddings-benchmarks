from web.embedding import Embedding
from web.vocabulary import *


def test_noinplace_transformword():
    cw = CountedVocabulary({' cat': 10, 'cat': 54, 'dog': 55})

    e = Embedding(cw, [[0, 0, 11], [0, 0, 12], [0, 0, 13]])
    e.transform_words(lambda x: x, inplace=False)


