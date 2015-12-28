# -*- coding: utf-8 -*-

"""
 Tests for similarity solvers
"""
from web.datasets.utils import _fetch_file
from web.embedding import Embedding
from web.datasets.similarity import fetch_SimLex999
from web.evaluate import evaluate_similarity

def test_similarity():
    url = "https://www.dropbox.com/s/rm756kjvckxa5ol/top100-sgns-googlenews-300.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    data = fetch_SimLex999()

    result_1 = evaluate_similarity(w, data.X, data.y)
    result_2 =  evaluate_similarity(dict(zip(w.vocabulary.words, w.vectors)), data.X, data.y)

    assert result_2 > 0
    assert result_1 == result_2, "evaluate_similarity should return same result for dict and Embedding instance"

