# -*- coding: utf-8 -*-

"""
 Tests for similarity solvers
"""
from web.datasets.utils import _fetch_file
from web.embedding import Embedding
from web.datasets.similarity import fetch_simlex999
from web.similarity import evaluate_similarity


def test_similarity():
    url = "https://www.dropbox.com/s/rm756kjvckxa5ol/top100-sgns-googlenews-300.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    data = fetch_simlex999()
    assert evaluate_similarity(w, data.X, data.y) > 0
