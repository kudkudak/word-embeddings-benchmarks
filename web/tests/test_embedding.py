# -*- coding: utf-8 -*-

"""
 Tests for embedding
"""
import tempfile
from web.datasets.utils import _fetch_file
from web.embedding import Embedding
from web.datasets.analogy import fetch_google_analogy
from web.datasets.similarity import fetch_SimLex999
from web.similarity import evaluate_similarity
from web.evaluate import evaluate_analogy, evaluate_on_semeval_2012_2, evaluate_on_WordRep
from web.utils import standardize_string
import numpy as np
from os import path

def test_standardize():
    url = "https://www.dropbox.com/s/rm756kjvckxa5ol/top100-sgns-googlenews-300.bin?dl=1"
    file_name = _fetch_file(url, "test")

    w = Embedding.from_word2vec(file_name, binary=True)
    w2 = w.standardize_words(inplace=False, lower=False, clean_words=True)
    w3 = Embedding.from_word2vec(file_name, binary=True)
    assert len(w2.words) == 95
    for word in w.vocabulary.words:
        if standardize_string(word, lower=False, clean_words=True):
            assert np.array_equal(w[word], w2[standardize_string(word, lower=False, clean_words=True)])

    w3.standardize_words(inplace=True, clean_words=True, lower=False)
    assert len(w3.words) == 95
    for word in w.vocabulary.words:
        if standardize_string(word, lower=False):
            assert np.array_equal(w[word], w3[standardize_string(word, lower=False, clean_words=True)])


def test_standardize_preserve_identity():
    d = {"Spider": [3, 4, 5], "spider": [1, 2, 3], "spideR": [3, 2, 4]}
    w3 = Embedding.from_dict(d)
    w4 = w3.standardize_words(inplace=False, lower=True)
    assert w4['spider'][0] == 1
    w3.standardize_words(inplace=True, lower=True)
    assert w3['spider'][0] == 1

def test_save():
    url = "https://www.dropbox.com/s/5occ4p7k28gvxfj/ganalogy-sg-wiki-en-400.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)

    dirpath = tempfile.mkdtemp()
    w.to_word2vec(w, path.join(dirpath, "tmp.bin"), binary=True)
    w.to_word2vec(w, path.join(dirpath, "tmp.txt"), binary=False)
    w2 = Embedding.from_word2vec(path.join(dirpath, "tmp.bin"), binary=True)
    w3 = Embedding.from_word2vec(path.join(dirpath, "tmp.txt"), binary=False)
    assert np.array_equal(w.vectors, w2.vectors)
    assert w.vocabulary.words == w2.vocabulary.words
    assert np.sum(np.abs(w.vectors - w3.vectors)) < 1e-5
    assert w.vocabulary.words == w3.vocabulary.words
