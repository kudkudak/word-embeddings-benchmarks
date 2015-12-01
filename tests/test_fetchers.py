# -*- coding: utf-8 -*-

"""
 Tests for data fetchers
"""

from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2, \
    fetch_wordrep
from web.datasets.similarity import fetch_simlex999

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
