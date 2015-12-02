# -*- coding: utf-8 -*-

"""
 Tests for data fetchers
"""

from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2, \
    fetch_wordrep
from web.datasets.similarity import fetch_simlex999, fetch_WS353, fetch_multilingual_simlex999, \
    fetch_MEN, fetch_MTurk, fetch_RW, fetch_RG65

from itertools import product


def test_MTurk_fetcher():
    data = fetch_MTurk()
    assert (len(data.y) == len(data.X) == 287)
    assert (10.0 >= data.y.max() >= 9)


def test_RW_fetcher():
    data = fetch_RW()
    assert (len(data.y) == len(data.X) == 2034)
    assert (10.0 >= data.y.max() >= 9.8)


def test_RG65_fetcher():
    data = fetch_RG65()
    assert (len(data.y) == len(data.X) == 65)
    assert (10.0 >= data.y.max() >= 9.8)


def test_MEN_fetcher():
    params = product(["all", "dev", "test"], ["natural", "lem"])
    data, V = {}, {}
    for which, form in params:
        fetched = fetch_MEN(which=which, form=form)
        data[which + ":" + form] = fetched
        V[which + ":" + form] = set([" ".join(sorted(x)) for x in data[which + ":" + form].X])
        assert fetched.y.max() <= 10.0

    assert V["dev:natural"].union(V["test:natural"]) == V["all:natural"]
    assert V["dev:lem"].union(V["test:lem"]) == V["all:lem"]
    assert data['all:natural']


def test_ws353_fetcher():
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


def test_simlex999_fetchers():
    data = fetch_simlex999()
    assert data.X.shape == (999, 2)

    for lang in ["EN", "RU", "IT", "DE"]:
        data = fetch_multilingual_simlex999(which=lang)
        assert data.y.shape[0] == data.sd.shape[0]
        assert data.X.shape[0] == 999


def test_analogy_fetchers():
    data = fetch_msr_analogy()
    assert len(data.y) == len(data.X) == 16

    data = fetch_google_analogy()
    assert len(data.y) == len(data.X) == 14
    assert len(data.categories_high_level) == 2

    data = fetch_semeval_2012_2()
    assert len(data.X) == len(data.y) == 79

    data = fetch_wordrep()
    assert len(data.categories_high_level) == 24
