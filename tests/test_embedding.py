# -*- coding: utf-8 -*-

"""
 Tests for embedding
"""

from web.datasets.utils import _fetch_files, _get_dataset_dir
from web.embedding import Embedding
import numpy as np

def test_standardize():
    data_dir = _get_dataset_dir('test')
    url = "https://www.dropbox.com/s/t1o1xi2v5m8znos/top100_word2vec.pkl?dl=1"
    file_name = _fetch_files(data_dir, [("top100_word2vec.pkl", url, {})], verbose=0)[0]
    w = Embedding.load(file_name)
    w2 = w.standardize_words(inplace=False)
    assert len(w2.words) == 95
    for word in w2.vocabulary.words:
        assert np.array_equal(w[word], w2[word])