# -*- coding: utf-8 -*-

"""
 Tests for embedding
"""

from web.datasets.utils import _fetch_file
from web.embedding import Embedding
from web.datasets.analogy import fetch_google_analogy
from web.analogy import evaluate_on_analogy, evaluate_on_semeval_2012_2, evaluate_on_WordRep
from web.utils import standardize_string
import numpy as np



def test_standardize():
    url = "https://www.dropbox.com/s/rm756kjvckxa5ol/top100-sgns-googlenews-300.bin?dl=1"
    file_name = _fetch_file(url, "test")

    w = Embedding.from_word2vec(file_name, binary=True)
    w2 = w.standardize_words(inplace=False)
    w3 = Embedding.from_word2vec(file_name, binary=True)
    assert len(w2.words) == 95
    for word in w.vocabulary.words:
        if standardize_string(word, lower=False):
            assert np.array_equal(w[word], w2[standardize_string(word, lower=False)])

    w3.standardize_words(inplace=True)
    assert len(w3.words) == 95
    for word in w.vocabulary.words:
        if standardize_string(word, lower=False):
            assert np.array_equal(w[word], w3[standardize_string(word, lower=False)])

def test_standardize_preserve_identity():
    d = {"Spider": [3,4,5], "spider": [1,2,3], "spideR": [3,2,4]}
    w3 = Embedding.from_dict(d)
    w4 = w3.standardize_words(inplace=False, lower=True)
    assert w4['spider'][0] == 1
    w3.standardize_words(inplace=True, lower=True)
    assert w3['spider'][0] == 1

def test_semeval_solver():
    url = "https://www.dropbox.com/s/rm756kjvckxa5ol/top100-sgns-googlenews-300.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    results = evaluate_on_semeval_2012_2(w)
    assert results['all'] >= 0, "Should have some results on SemEval2012"


def test_wordrep_solver():
    url = "https://www.dropbox.com/s/5occ4p7k28gvxfj/ganalogy-sg-wiki-en-400.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    P = evaluate_on_WordRep(w, max_pairs=2)
    assert P['accuracy']['all'] >= 0


def test_analogy_solver():
    url = "https://www.dropbox.com/s/5occ4p7k28gvxfj/ganalogy-sg-wiki-en-400.bin?dl=1"
    file_name = _fetch_file(url, "test")

    w = Embedding.from_word2vec(file_name, binary=True)
    data = fetch_google_analogy()
    ids = np.random.RandomState(777).choice(range(data.X.shape[0]), 1000, replace=False)
    X, y = data.X[ids], data.y[ids]
    category = data.category_high_level[ids]

    results = evaluate_on_analogy(w=w, X=X, y=y, category=category)
    assert results['accuracy']['all'] >= 0.65
    assert results['accuracy']['semantic'] >= 0.7
    assert results['accuracy']['syntactic'] >= 0.63

    results = evaluate_on_analogy(w=w, X=X, y=y, category=category, method="mul")
    assert results['accuracy']['all'] >= 0.7
    assert results['accuracy']['semantic'] >= 0.75
    assert results['accuracy']['syntactic'] >= 0.64

    results_mul = evaluate_on_analogy(w=w, X=X, y=y, category=category, method="mul", k=400)
    results_add = evaluate_on_analogy(w=w, X=X, y=y, category=category, method="add", k=400)
    assert results_mul['accuracy']['all'] >= results_add['accuracy']['all']
    assert results_mul['accuracy']['syntactic'] >= results_add['accuracy']['syntactic']
    assert results_mul['accuracy']['semantic'] >= results_add['accuracy']['semantic']


def test_save(tmpdir):
    url = "https://www.dropbox.com/s/5occ4p7k28gvxfj/ganalogy-sg-wiki-en-400.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    w.to_word2vec(w, tmpdir.mkdir("web").join("tmp.bin"), binary=True)
    w.to_word2vec(w, tmpdir.mkdir("web").join("tmp.txt"), binary=True)
    w2 = Embedding.from_word2vec(tmpdir.mkdir("web").join("tmp.bin"), binary=True)
    w3 = Embedding.from_word2vec(tmpdir.mkdir("web").join("tmp.txt"), binary=False)
    assert np.array_equal(w.vectors, w2.vectors)
    assert w.vocabulary.words == w2.vocabulary.words
    assert np.array_equal(w.vectors, w3.vectors)
    assert w.vocabulary.words == w3.vocabulary.words