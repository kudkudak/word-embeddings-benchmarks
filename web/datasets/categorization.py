# -*- coding: utf-8 -*-

"""
 Functions for fetching categorization datasets
"""

from sklearn.datasets.base import Bunch
from .utils import _get_cluster_assignments


def fetch_AP():
    """
    Fetch Almuhareb and Abdulrahman categorization dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'clusters': dict of arrays of words representing

    References
    ----------
    Almuhareb et al., "Concept learning and categorization from the web", 2005

    Notes
    -----
    Authors description:
    Our goal was to create a dataset balanced with respect to
    three factors: class type, frequency, and ambiguity.
    First of all, we aimed to include one class of nouns for
    each of the 21 unique beginners of the WordNet noun
    hierarchy4
    . We chose subclasses for each of these 21
    beginners that would represent a reasonably natural cluster:
    e.g., the hyponym social occasion for the unique beginner
    event. From each such class, we selected between 13 and 21
    nouns to be representative concepts for the class (e.g.,
    ceremony, feast, and graduation for the class social
    occasion).
    Secondly, we aimed to include about 1/3 high frequency
    nouns, 1/3 medium frequency, and 1/3 low frequency. Noun
    frequencies where estimated using the British National
    Corpus. We considered as highly frequent those nouns with
    frequency 1,000 or more; as medium frequent the nouns
    with between 1,000 and 100 occurrences; and those between
    100 and 5 as low frequent.
    Thirdly, we wanted the dataset to be balanced as to
    ambiguity, estimated on the basis of the number of senses in
    WordNet. Nouns with 4 or more senses were considered
    highly ambiguous; nouns with 2 or 3 senses medium
    ambiguous; and nouns with a single sense as not ambiguous.
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

    References
    ----------
    Baroni et al. "How we BLESSed distributional semantic evaluation", 2011

    Notes
    -----
    Data set includes 200 concrete nouns (100 animate and 100 inanimate nouns)
    from different classes (e.g., tools, clothing, vehicles, animals, etc.).
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

    This is not the same dataset as 'battig' in Baroni et al. "Don’t count, predict! A systematic comparison of
    context-counting vs. context-predicting semantic vectors"
    """
    data = _get_cluster_assignments(dataset_name="EN-BATTIG",
                                    url="https://www.dropbox.com/sh/ckp4yu7k7xl7u2a/AABhmpgU3ake3T9liA9BR8EBa?dl=1",
                                    sep=",", skip_header=True)
    return Bunch(X=data.X[:, 0], y=data.y,
                 freq=data.X[:, 1], frequency=data.X[:, 2], rank=data.X[:, 3], rfreq=data.X[:, 4])



def fetch_ESSLI_2c():
    """
    Fetch ESSLI 2c task categorization dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment

    References
    ----------
    Originally published at http://wordspace.collocations.de/doku.php/data:esslli2008:verb_categorization

    Notes
    -----
    The goal of the sub-task is to group verbs into semantic categories. The data set consists of 45 verbs,
    belonging to 9 semantic classes. The classification scheme is inspired by P. Vinson & G. Vigliocco (2007),
    “Semantic Feature Production Norms for a Large Set of Objects and Events”, Behavior Research Methods,
    which in turn closely follows the classification proposed in Levin (1993). The data set consists of 44 concrete
    nouns, belonging to 6 semantic categories (four animates and two inanimates). The nouns are included in the
    feature norms described in McRae et al. (2005)
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

    References
    ----------
    Originally published at
    http://wordspace.collocations.de/doku.php/data:esslli2008:abstract_concrete_nouns_discrimination.

    Notes
    -----
    The data set consists of 40 nouns extracted from the MRC Psycholinguistic Database, with ratings by human subjects
    on the concreteness scale. The nouns have been classified into three classes: HI, LO and ME being highly,
    low and medium abstract nouns.
    """
    return _get_cluster_assignments(dataset_name="EN-ESSLI-2b",
                                    url="https://www.dropbox.com/sh/7gdv52gy9vb4mf2/AACExLgHdbvbBrRZBP6CcdDaa?dl=1")


def fetch_ESSLI_1a():
    """
    Fetch ESSLI 1a task categorization dataset.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': words
        'y': cluster assignment

    References
    ----------
    Originally published at http://wordspace.collocations.de/doku.php/data:esslli2008:concrete_nouns_categorization.

    Notes
    -----
    The goal of the sub-task is to group concrete nouns into semantic categories.
    The data set consists of 44 concrete nouns, belonging to 6 semantic categories (four animates and two inanimates).
    The nouns are included in the feature norms described in McRae et al. (2005)
    """
    return _get_cluster_assignments(dataset_name="EN-ESSLI-1a",
                                    url="https://www.dropbox.com/sh/h362565r1sk5wii/AADjcdYy3nRo-MjuFUSvb-0ya?dl=1")
