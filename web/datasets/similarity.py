# -*- coding: utf-8 -*-

"""
 Functions for fetching similarity data
"""

import pandas as pd

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_files

def fetch_simlex999():
    """
    Fetch SimLex999 dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,
        'conc': matrix with columns conc(w1), conc(w2) and concQ the from dataset
        'POS': vector with POS tag
        'assoc': matrix with columns denoting free association: Assoc(USF) and SimAssoc333

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """

    dataset_name = 'analogy/EN-SIM999'
    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    url = "https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-SIM999.txt", url, {})],
                            resume=True,
                            verbose=0)[0]

    data = pd.read_csv(raw_data, "\t")
    # We basically select all the columns available
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    return Bunch(X=X, y=y, sd=sd, conc=conc, POS=POS, assoc=assoc)
