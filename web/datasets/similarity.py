# -*- coding: utf-8 -*-

"""
 Functions for fetching similarity datasets
"""

import numpy as np

from sklearn.datasets.base import Bunch
from .utils import _get_as_pd

def fetch_MTurk():
    """
    Fetch MTurk dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    Scores were scaled by factor of 2

    """
    data = _get_as_pd('https://www.dropbox.com/s/f1v4ve495mmd9pw/EN-TRUK.txt?dl=1',
                      'similarity', header=None, sep=" ").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=2 * data[:, 2].astype(np.float))


def fetch_MEN(which="all", form="natural"):
    """
    Fetch MEN dataset for testing similarity and relatedness

    Parameters
    -------
    which : "all", "test" or "dev"
    form : "lem" or "natural"

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    Scores for MEN are calculated differently than in WS353 or SimLex999.
    Furthermore scores where rescaled to 0 - 10 scale to match standard scaling

    """

    if which == "dev":
        data = _get_as_pd('https://www.dropbox.com/s/c0hm5dd95xapenf/EN-MEN-LEM-DEV.txt?dl=1',
                          'similarity', header=None, sep=" ")
    elif which == "test":
        data = _get_as_pd('https://www.dropbox.com/s/vdmqgvn65smm2ah/EN-MEN-LEM-TEST.txt?dl=1',
                          'similarity/EN-MEN-LEM-TEST', header=None, sep=" ")
    elif which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/b9rv8s7l32ni274/EN-MEN-LEM.txt?dl=1',
                          'similarity', header=None, sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    if form == "natural":
        # Remove last two chars from first two columns
        data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    elif form != "lem":
        raise RuntimeError("Not recognized form argument")

    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(np.float) / 5.0)


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
    if which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "relatedness":
        data = _get_as_pd('https://www.dropbox.com/s/x94ob9zg0kj67xg/EN-WSR353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "similarity":
        data = _get_as_pd('https://www.dropbox.com/s/ohbamierd2kt1kp/EN-WSS353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "set1":
        data = _get_as_pd('https://www.dropbox.com/s/opj6uxzh5ov8gha/EN-WS353-SET1.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "set2":
        data = _get_as_pd('https://www.dropbox.com/s/w03734er70wyt5o/EN-WS353-SET2.txt?dl=1',
                          'similarity', header=0, sep="\t")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(np.float)

    # We have also scores
    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(np.float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    else:
        return Bunch(X=X.astype("object"), y=y)


def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity

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
    Scores were scaled by factor 10/4

    """
    data = _get_as_pd('https://www.dropbox.com/s/chopke5zqly228d/EN-RG-65.txt?dl=1',
                      'similarity', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def fetch_RW():
    """
    Fetch Rare Words dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """
    data = _get_as_pd('https://www.dropbox.com/s/xhimnr51kcla62k/EN-RW.txt?dl=1',
                      'similarity', header=None, sep="\t").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float),
                 sd=np.std(data[:, 3:].astype(np.float)))


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
        data = _get_as_pd('https://www.dropbox.com/s/nczc4ao6koqq7qm/EN-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "DE":
        data = _get_as_pd('https://www.dropbox.com/s/ucpwrp0ahawsdtf/DE-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "IT":
        data = _get_as_pd('https://www.dropbox.com/s/siqjagyz8dkjb9q/IT-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "RU":
        data = _get_as_pd('https://www.dropbox.com/s/3v26edm9a31klko/RU-MSIM999.txt?dl=1',
                          'similaerity', header=None, encoding='utf-8', sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    scores = data.values[:, 2:].astype(np.float)
    y = np.mean(scores, axis=1)
    sd = np.std(scores, axis=1)

    return Bunch(X=X.astype("object"), y=y, sd=sd)


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

    data = _get_as_pd('https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1',
                      'similarity', sep="\t")

    # We basically select all the columns available
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    return Bunch(X=X.astype("object"), y=y, sd=sd, conc=conc, POS=POS, assoc=assoc)
