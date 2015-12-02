# -*- coding: utf-8 -*-

"""
 Tests for data fetchers
"""

from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2, \
    fetch_wordrep
from web.datasets.similarity import fetch_simlex999, fetch_WS353

def test_similarity_fetchers():
    data1 = fetch_WS353(which="set1")
    data2 = fetch_WS353(which="set2")
    data3 = fetch_WS353(which="similarity")
    data4 = fetch_WS353(which="relatedness")
    data5 = fetch_WS353(which="all")
    V5 = set([" ".join(sorted(x)) for x in data5.X])
    V1 = set([" ".join(sorted(x)) for x in data1.X])
    V2 = set([" ".join(sorted(x)) for x in data2.X])
    V3 = set([" ".join(sorted(x)) for x in data3.X])
    V4 = set([" ".join(sorted(x)) for x in data4.X])

    # sd and scores have same length
    assert data1.sd.shape[0] == data1.y.shape[0]
    assert data2.sd.shape[0] == data2.y.shape[0]

    # WSR = WSR-SET1 u WSR-SET2
    assert data5.X.shape[0] == 353
    assert V5 == V2.union(V1)

    assert V5 == V3.union(V4)

    # Two word pairs reoccurr
    assert len(V5) == 351

def test_analogy_fetchers():
    data = fetch_msr_analogy()
    assert len(data.y) == len(data.X) == 16

    data = fetch_google_analogy()
    assert len(data.y) == len(data.X) == 14
    assert len(data.categories_high_level) == 2

    data = fetch_semeval_2012_2()
    assert len(data.X) == len(data.y) ==  79

    data = fetch_wordrep()
    assert len(data.categories_high_level) == 24

    data = fetch_simlex999()
    assert data.X.shape == (999, 2)
