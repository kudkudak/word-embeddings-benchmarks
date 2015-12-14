# -*- coding: utf-8 -*-

"""
 Tests for embedding
"""

from web.datasets.utils import _fetch_file
from web.embedding import Embedding
import numpy as np

def test_standardize():
    url = "https://www.dropbox.com/s/t1o1xi2v5m8znos/top100_word2vec.pkl?dl=1"
    file_name = _fetch_file(url)

    w = Embedding.load(file_name)
    w2 = w.standardize_words(inplace=False)
    w3 = Embedding.load(file_name)
    assert len(w2.words) == 95
    for word in w2.vocabulary.words:
        assert np.array_equal(w[word], w2[word])

    w3.standardize_words(inplace=True)
    assert len(w3.words) == 95
    for word in w3.vocabulary.words:
        assert np.array_equal(w[word], w3[word])


# TODO: add test for analogy and category
def test_analogy_solver():
    pass
    # a) with k
    # b) without k

# TODO: add test for save
def test_save():
    pass