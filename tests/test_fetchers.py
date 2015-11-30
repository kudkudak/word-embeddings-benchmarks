# -*- coding: utf-8 -*-

"""
 Tests for data fetchers
"""

import pytest
from web.datasets import fetch_google_analogy, fetch_msr_analogy

def test_analogy_fetchers():
    data = fetch_msr_analogy()
    assert len(data.answers) == len(data.questions) == 16

    data = fetch_google_analogy()
    assert len(data.answers) == len(data.questions) == 14
    assert len(data.categories_high_level) == 2
