# -*- coding: utf-8 -*-

"""
 Functions for fetching similarity data
"""

import pandas as pd
import numpy as np

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_files

def fetch_WS353(which="all"):
    """
    Fetch WS353 dataset for testing attributional and
    relatedness similarity

    Parameters
    -------
    which : 'all': for both relatedness and attributional similarity,
            'relatedness': for relatedness similarity
            'similarity': for attributional similarity
            'set1': as divided by authors
            'set2': as divided by authors

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """
    header = None
    if which == "all":
        header = 0
        dataset_name = 'similarity/EN-WS353'
        url = "https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1"
        file_name = "EN-WS353.txt"
    elif which == "relatedness":
        dataset_name = 'similarity/EN-WSR353'
        url = "https://www.dropbox.com/s/x94ob9zg0kj67xg/EN-WSR353.txt?dl=1"
        file_name = "EN-WSR353.txt"
    elif which == "similarity":
        dataset_name = 'similarity/EN-WSS353'
        file_name = "EN-WSS353.txt"
        url = "https://www.dropbox.com/s/ohbamierd2kt1kp/EN-WSS353.txt?dl=1"
    elif which == "set1":
        header = 0
        dataset_name = 'similarity/EN-WS353-SET1'
        file_name = "EN-WS353-SET1.txt"
        url = "https://www.dropbox.com/s/opj6uxzh5ov8gha/EN-WS353-SET1.txt?dl=1"
    elif which == "set2":
        header = 0
        dataset_name = 'similarity/EN-WS353-SET2'
        file_name = "EN-WS353-SET2.txt"
        url = "https://www.dropbox.com/s/w03734er70wyt5o/EN-WS353-SET2.txt?dl=1"
    else:
        raise RuntimeError("Not recognized which parameter")

    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    raw_data = _fetch_files(data_dir, [(file_name, url, {})],
                            resume=True,
                            verbose=0)[0]
    data = pd.read_csv(raw_data, "\t", header=header)

    # We basically select all the columns available
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(np.float)

    # We have also scores
    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(np.float), axis=1).flatten()
        return Bunch(X=X, y=y, sd=sd)
    else:
        return Bunch(X=X, y=y)


def fetch_multilingual_simlex999(which="EN"):
    """
    Fetch Multilingual SimLex999 dataset for testing attributional similarity

    Parameters
    -------
    which : "EN", "RU", "IT" or "DE" for language

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    Scores for EN are different than the original SimLex999 dataset

    """

    if which == "EN":
        dataset_name = 'similarity/EN-MSIM999'
        url = "https://www.dropbox.com/s/nczc4ao6koqq7qm/EN-MSIM999.txt?dl=1"
        file_name = "EN-MSIM999.txt"
    elif which == "DE":
        dataset_name = 'similarity/DE-MSIM999'
        url = "https://www.dropbox.com/s/ucpwrp0ahawsdtf/DE-MSIM999.txt?dl=1"
        file_name = "DE-MSIM999.txt"
    elif which == "IT":
        dataset_name = 'similarity/IT-MSIM999'
        file_name = "IT-MSIM999.txt"
        url = "https://www.dropbox.com/s/siqjagyz8dkjb9q/IT-MSIM999.txt?dl=1"
    elif which == "RU":
        dataset_name = 'similarity/RU-MSIM999'
        file_name = "RU-MSIM999.txt"
        url = "https://www.dropbox.com/s/3v26edm9a31klko/RU-MSIM999.txt?dl=1"
    else:
        raise RuntimeError("Not recognized which parameter")

    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    raw_data = _fetch_files(data_dir, [(file_name, url, {})],
                            resume=True,
                            verbose=0)[0]
    data = pd.read_csv(raw_data, " ", encoding='utf-8', header=None)

    # We basically select all the columns available
    X = data.values[:, 0:2]
    scores = data.values[:, 2:].astype(np.float)
    y = np.mean(scores, axis=1)
    sd = np.std(scores, axis=1)

    return Bunch(X=X, y=y, sd=sd)

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
