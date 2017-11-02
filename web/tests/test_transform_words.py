from web.embedding import Embedding
from web.vocabulary import *

import numpy as np
import logging
import sys


# COUNTEDVOCABULARY

def test_noinplace_transform_word_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(word_count=[(' cat ', 10), ('cat', 50), ('dog', 60)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 11], [0, 11, 12], [0, 12, 13]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 2
    assert len(pe.vectors) == 2

    # 'dog'
    assert [0, 0, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 11])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary


def test_noinplace_transform_word_prefer_occurences_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(word_count=[(' cat ', 5), ('pikatchu ', 10), ('cat', 50), ('dog', 60), ('pikatchu', 200)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # 'dog'
    assert [0, 1, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()
    # pikatchu
    assert [0, 0, 1] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[0] == 'pikatchu'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    assert d['pikatchu'] == 200
    # dog
    assert pe.vocabulary.words[1] == 'dog'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[2] == 'cat'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary


def test_noinplace_transform_word_prefer_shortestword_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(
        word_count=[('dog', 60), ('cat', 50), ('    pikatchu   ', 10), ('pikatchu', 10), (' cat ', 5)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 0, 1] in pe.vectors.tolist()
    # 'cat'
    assert [0, 1, 11] in pe.vectors.tolist()
    # pikatchu
    assert [0, 12, 13] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 12, 13])
    assert d['pikatchu'] == 10

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary


# ORDERDVOCABULARY

def test_noinplace_transform_word_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['dog', 'cat', '  cat'])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 11], [0, 11, 12], [0, 12, 13]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 2
    assert len(pe.vectors) == 2

    # 'dog'
    assert [0, 0, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 11])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])

    assert type(pe.vocabulary) == OrderedVocabulary


def test_noinplace_transform_word_prefer_occurences_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['pikatchu', 'dog', 'cat', 'pikatchu ', ' cat '])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 1, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()
    # pikatchu
    assert [0, 0, 1] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[0] == 'pikatchu'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    # dog
    assert pe.vocabulary.words[1] == 'dog'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])

    # cat
    assert pe.vocabulary.words[2] == 'cat'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])

    assert type(pe.vocabulary) == OrderedVocabulary


def test_noinplace_transform_word_prefer_shortestword_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['dog', 'cat', '    pikatchu   ', 'pikatchu', ' cat  '])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 0, 1] in pe.vectors.tolist()
    # 'cat'
    assert [0, 1, 11] in pe.vectors.tolist()
    # pikatchu
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])

    assert type(pe.vocabulary) == OrderedVocabulary


# VOCABULARY

def test_noinplace_transform_word_Vocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = Vocabulary(words=['dog', 'cat', '  cat '])
    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 11], [0, 11, 12], [0, 12, 13]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 2
    assert len(pe.vectors) == 2

    # 'dog'
    assert [0, 0, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 11])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])

    assert type(pe.vocabulary) == Vocabulary


def test_noinplace_transform_word_prefer_shortest_ord1_Vocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = Vocabulary(words=['pikatchu ', 'dog', 'cat', 'pikatchu', '  cat '])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 12, 13], [0, 1, 11], [0, 11, 12], [0, 0, 1], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 1, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()
    # pikatchu
    assert [0, 0, 1] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 0, 1])

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 1, 11])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])

    assert type(pe.vocabulary) == Vocabulary


def test_noinplace_transform_word_prefer_shortestword2_Vocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = Vocabulary(words=['dog', 'cat', '    pikatchu   ', 'pikatchu', ' cat '])
    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=False)

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 0, 1] in pe.vectors.tolist()
    # 'cat'
    assert [0, 1, 11] in pe.vectors.tolist()
    # pikatchu
    assert [0, 12, 13] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 12, 13])

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])

    assert type(pe.vocabulary) == Vocabulary

####################### inplace= True #######################

# COUNTEDVOCABULARY

def test_inplace_transform_word_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(word_count=[(' cat ', 10), ('cat', 50), ('dog', 60)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 11], [0, 11, 12], [0, 12, 13]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 2
    assert len(pe.vectors) == 2

    # 'dog'
    assert [0, 0, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 11])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary


def test_inplace_transform_word_prefer_occurences_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(word_count=[(' cat ', 5), ('pikatchu ', 10), ('cat', 50), ('dog', 60), ('pikatchu', 200)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # 'dog'
    assert [0, 1, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()
    # pikatchu
    assert [0, 0, 1] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[0] == 'pikatchu'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    assert d['pikatchu'] == 200
    # dog
    assert pe.vocabulary.words[1] == 'dog'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[2] == 'cat'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary


def test_inplace_transform_word_prefer_shortestword_CountedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = CountedVocabulary(
        word_count=[('dog', 60), ('cat', 50), ('    pikatchu   ', 10), ('pikatchu', 10), (' cat ', 5)])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 0, 1] in pe.vectors.tolist()
    # 'cat'
    assert [0, 1, 11] in pe.vectors.tolist()
    # pikatchu
    assert [0, 12, 13] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    l = pe.vocabulary.getstate()
    d = {l[0][i]: l[1][i] for i in range(len(l[0]))}

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 12, 13])
    assert d['pikatchu'] == 10

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    assert d['dog'] == 60

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])
    assert d['cat'] == 50

    assert type(pe.vocabulary) == CountedVocabulary

# ORDERDVOCABULARY

def test_inplace_transform_word_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['dog', 'cat', '  cat'])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 11], [0, 11, 12], [0, 12, 13]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 2
    assert len(pe.vectors) == 2

    # 'dog'
    assert [0, 0, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 11])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 11, 12])

    assert type(pe.vocabulary) == OrderedVocabulary


def test_inplace_transform_word_prefer_occurences_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['pikatchu', 'dog', 'cat', 'pikatchu ', ' cat '])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 1, 11] in pe.vectors.tolist()
    # 'cat'
    assert [0, 11, 12] in pe.vectors.tolist()
    # pikatchu
    assert [0, 0, 1] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[0] == 'pikatchu'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])
    # dog
    assert pe.vocabulary.words[1] == 'dog'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])

    # cat
    assert pe.vocabulary.words[2] == 'cat'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])

    assert type(pe.vocabulary) == OrderedVocabulary


def test_inplace_transform_word_prefer_shortestword_OrderedVocabulary():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    cw = OrderedVocabulary(words=['dog', 'cat', '    pikatchu   ', 'pikatchu', ' cat  '])

    e = Embedding(vocabulary=cw, vectors=np.asanyarray([[0, 0, 1], [0, 1, 11], [0, 11, 12], [0, 12, 13], [0, 13, 14]]))
    pe = e.transform_words(lambda x: x.strip(), inplace=True)

    assert pe is e and pe == e

    assert len(pe.vocabulary) == 3
    assert len(pe.vectors) == 3

    # 'dog'
    assert [0, 0, 1] in pe.vectors.tolist()
    # 'cat'
    assert [0, 1, 11] in pe.vectors.tolist()
    # pikatchu
    assert [0, 11, 12] in pe.vectors.tolist()

    assert 'cat' in pe.vocabulary.words
    assert 'dog' in pe.vocabulary.words
    assert 'pikatchu' in pe.vocabulary.words

    # pikatchu
    assert pe.vocabulary.words[2] == 'pikatchu'
    assert np.array_equal(pe.vectors[2], [0, 11, 12])

    # dog
    assert pe.vocabulary.words[0] == 'dog'
    assert np.array_equal(pe.vectors[0], [0, 0, 1])

    # cat
    assert pe.vocabulary.words[1] == 'cat'
    assert np.array_equal(pe.vectors[1], [0, 1, 11])

    assert type(pe.vocabulary) == OrderedVocabulary


