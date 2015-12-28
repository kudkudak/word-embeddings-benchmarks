import numpy as np
from web.evaluate import calculate_purity, evaluate_categorization
from web.embedding import Embedding
from web.datasets.utils import _fetch_file
from web.datasets.categorization import fetch_ESSLI_2c

def test_purity():
    y_true = np.array([1,1,2,2,3])
    y_pred = np.array([2,2,2,2,1])
    assert abs(0.6 - calculate_purity(y_true, y_pred)) < 1e-10

def test_categorization():
    data = fetch_ESSLI_2c()
    url = "https://www.dropbox.com/s/5occ4p7k28gvxfj/ganalogy-sg-wiki-en-400.bin?dl=1"
    file_name = _fetch_file(url, "test")
    w = Embedding.from_word2vec(file_name, binary=True)
    assert evaluate_categorization(w, data.X, data.y, seed=777, method="all") >= 0.2