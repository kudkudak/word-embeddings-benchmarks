# -*- coding: utf-8 -*-

"""
 Functions for fetching categorization datasets
"""

from sklearn.datasets.base import Bunch
from .utils import _get_cluster_assignments

def fetch_AP():
    """
    Fetch Almuhareb and Abdulrahman categorization dataset

    Parameters
    -------

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'clusters': dict of arrays of words representing
    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: <>

    """
    return _get_cluster_assignments(dataset_name="EN-AP",
                                    url="https://www.dropbox.com/sh/6xu1c1aan8f83p3/AACMyoLwncNhRkUkqvGurYB6a?dl=1")


def fetch_BLESS():
    """
    Fetch Baroni and Marco categorization dataset

    Parameters
    -------

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment
        'names': cluster names

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: <>

    """
    return _get_cluster_assignments(dataset_name="EN-BLESS",
                                    url="https://www.dropbox.com/sh/5qbl5cmh17o3eh0/AACyCEqpMktdMI05zwphJRI7a?dl=1")


def fetch_battig():
    """
    Fetch 1969 Battig dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment
        'names': cluster names
        'freq': frequency of response
        'frequency': Kucera-Francis word frequency
        'rank': rank of frequence within response
        'rfreq': rated frequency

    References
    ----------
    W.F Battig & W.E Montague (1968). Category norms for verbal items in 56 categories: A replication
    and extension of the Connecticut norms using University of Maryland and Illinois students
    (Tech. Rep.) University of Colorado, Boulder, CO (1968)

    Notes
    -----
    This dataset comprises a ranked list of 5231 words listed in 56 taxonomic categories by people
    who were asked to list as many exemplars of a given category ("a precious stone", "a unit of time",
    "a fruit", "a color", etc.). Participants had 30s to generate as many responses to each category as
    possible, after which time the next category name was presented.
    Included in this dataset are all words from the Battig and Montague (1969) norms listed with
    freq > 1.

    """
    data = _get_cluster_assignments(dataset_name="EN-BATTIG",
                                    url="https://www.dropbox.com/sh/ckp4yu7k7xl7u2a/AABhmpgU3ake3T9liA9BR8EBa?dl=1",
                                    sep=",", skip_header=True)
    return Bunch(X=data.X[:, 0], y=data.y, names=data.names,
                 freq=data.X[:, 1], frequency=data.X[:, 2], rank=data.X[:, 3], rfreq=data.X[:, 4])


def fetch_ESSLI_2c():
    """
    Fetch ESSLI 2c task categorization dataset

    Parameters
    -------

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment
        'names': cluster names

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: <>

    """
    return _get_cluster_assignments(dataset_name="EN-ESSLI-2c",
                                    url="https://www.dropbox.com/sh/d3mcyl3b5mawfhm/AAABygW1rguhI4L0XSw_I68ta?dl=1")


def fetch_ESSLI_2b():
    """
    Fetch ESSLI 2c task categorization dataset

    Parameters
    -------

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment
        'names': cluster names

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: <>

    """
    return _get_cluster_assignments(dataset_name="EN-ESSLI-2b",
                                    url="https://www.dropbox.com/sh/7gdv52gy9vb4mf2/AACExLgHdbvbBrRZBP6CcdDaa?dl=1")


def fetch_ESSLI_1a():
    """
    Fetch ESSLI 1a task categorization dataset

    Parameters
    -------

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment
        'names': cluster names

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: <>

    """
    return _get_cluster_assignments(dataset_name="EN-ESSLI-1a",
                                    url="https://www.dropbox.com/sh/h362565r1sk5wii/AADjcdYy3nRo-MjuFUSvb-0ya?dl=1")
