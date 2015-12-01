"""
 Functions for fetching data
"""

from sklearn.datasets.base import Bunch
from sklearn.utils import check_random_state

import pandas as pd
import numpy as np

from .utils import _get_dataset_dir, _fetch_files, _change_list_to_np
from collections import defaultdict
import glob, os

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
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    return Bunch(X=X, y=y, sd=sd, conc=conc, POS=POS, assoc=assoc)

def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': dictionary keyed on category with word matrix of size N x 3
        'y': dictionary keyed on category with answers as vector of words

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """

    dataset_name = 'analogy/EN-MSR'
    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    url = "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-MSR.txt", url, {})],
                            resume=True,
                            verbose=0)[0]

    with open(raw_data, "r") as f:
        L = f.read().splitlines()

    questions = defaultdict(list)
    answers = defaultdict(list)
    for l in L:
        words = l.split()
        questions[words[3]].append(words[0:3])
        answers[words[3]].append(words[4])

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
        'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
        'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(X=_change_list_to_np(questions), y=_change_list_to_np(answers))

def fetch_semeval_2012_2(which="all", which_scoring="golden"):
    """
    Fetch dataset used for SEMEVAL 2012 task 2 competition

    Parameters
    -------
    which : "all", "train" or "test"
    which_scoring: "golden" or "platinium" (see Notes)

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X_prot': dictionary keyed on category. Each entry is a matrix of prototype word pairs (see Notes)
        'X': dictionary keyed on category. Each entry is a matrix of question word pairs
        'y': dictionary keyed on category. Each entry is a dictionary word pair -> score

        'categories_names': dictionary keyed on category. Each entry is a human readable name of
        category.
        'categories_descriptions': dictionary keyed on category. Each entry is a human readable description of
        category.

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    Dataset used in competition was scored as in golden scoring (which_scoring) parameter, however
    organiser have release improved labels afterwards (platinium scoring)


    """
    assert which in ['all', 'train', 'test']
    assert which_scoring in ['golden', 'platinium']

    data_dir = _get_dataset_dir("analogy", verbose=0)

    raw_data = _fetch_files(data_dir, [("EN-SEMVAL-2012-2",
                                        "https://www.dropbox.com/sh/yjzunhyqzsu1z47/AAAjyWDfP_ZAkmmNus4YBAEHa?dl=1",
                                        {'uncompress': True, "move": "EN-SEMVAL-2012-2/EN-SEMVAL-2012-2.zip"})],
                            verbose=0)[0]
    train_files = set(glob.glob(os.path.join(raw_data, "train*.txt"))) - \
        set(glob.glob(os.path.join(raw_data, "train*_meta.txt")))
    test_files = set(glob.glob(os.path.join(raw_data, "test*.txt"))) - \
        set(glob.glob(os.path.join(raw_data, "test*_meta.txt")))

    if which == "train":
        files = train_files
    elif which == "test":
        files = test_files
    elif which == "all":
        files = train_files.union(test_files)

    questions = defaultdict(list)
    prototypes = {}
    golden_scores = {}
    platinium_scores = {}
    scores = {"platinium": platinium_scores, "golden": golden_scores}
    categories_names = {}
    categories_descriptions = {}
    for f in files:
        with open(f[0:-4] + "_meta.txt") as meta_f:
            meta = meta_f.read().splitlines()[1].split(",")

        with open(os.path.dirname(f) + "/pl-" + os.path.basename(f)) as f_pl:
            platinium = f_pl.read().splitlines()

        with open(f) as f_gl:
            golden = f_gl.read().splitlines()

        assert platinium[0] == golden[0]

        c = meta[0] + "_" + meta[1]
        categories_names[c] = meta[2] + "_" + meta[3]
        categories_descriptions[c] = meta[4]

        prototypes[c] = [l.split(":") for l in platinium[0].split(",")]
        golden_scores[c] = {}
        platinium_scores[c] = {}

        for line_pl in platinium[1:]:
            word_pair, score = line_pl.split()
            questions[c].append(word_pair.split(":"))
            platinium_scores[c][word_pair] = score

        for line_g in golden[1:]:
            word_pair, score = line_g.split()
            golden_scores[c][word_pair] = score


    return Bunch(X_prot=_change_list_to_np(questions),
                 X=_change_list_to_np(questions),
                 y_=scores[which_scoring],
                 categories_names=categories_names,
                 categories_descriptions=categories_descriptions)

def fetch_wordrep(subsample=None, rng=None):
    """
    Fetch  MSR WordRep dataset for testing both syntactic and semantic dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'word_pairs': dictionary keyed on category with word matrix of words.
        You can form questions by taking any pair of pairs of words
        'categories_high_level': dictionary keyed on higher level category that
        provides coarse grained grouping of categories

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """
    data_dir = _get_dataset_dir("analogy", verbose=0)

    raw_data = _fetch_files(data_dir, [("EN-WORDREP",
                                        "https://www.dropbox.com/sh/5k78h9gllvc44vt/AAALLQq-Bge605OIMlmGBbNJa?dl=1",
                                        {'uncompress': True, "move": "EN-WORDREP/EN-WORDREP.zip"})],
                            verbose=0)[0]

    wikipedia_dict = glob.glob(raw_data + "/Pairs_from_Wikipedia_and_Dictionary/*.txt")
    wordnet = glob.glob(raw_data + "/Pairs_from_WordNet/*.txt")

    word_pairs = defaultdict(list)
    files = wikipedia_dict + wordnet
    categories_high_level = {}
    for f in files:
        c = os.path.basename(f).split(".")[0].split("-")[1]
        with open(wikipedia_dict[0], "r") as f:
            for l in f.read().splitlines():
                word_pairs[c].append(l.split())

    if subsample:
        assert subsample <= 1.0
        rng = check_random_state(rng)
        for c in word_pairs:
            ids = rng.choice(range(len(word_pairs[c])), int(subsample*len(word_pairs[c])), replace=False)
            word_pairs[c] = [word_pairs[c][i] for i in ids]

    for f in wikipedia_dict:
        c = os.path.basename(f).split(".")[0].split("-")[1]
        categories_high_level[c] = "wikipedia-dict"

    for f in wordnet:
        c = os.path.basename(f).split(".")[0].split("-")[1]
        categories_high_level[c] = "wordnet"

    return Bunch(categories_high_level=categories_high_level,
                 word_pairs=_change_list_to_np(word_pairs))

def fetch_google_analogy():
    """
    Fetch Google dataset for testing both semantic and syntactic analogies.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': dictionary keyed on category with matrix of word questions
        'y': dictionary keyed on category with the answer word
        'categories_high_level': dictionary keyed on higher level category that
        provides coarse grained grouping of categories

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Superseded by WordRep dataset.

    """

    dataset_name = 'analogy/EN-GOOGLE'
    data_dir = _get_dataset_dir(dataset_name, data_dir=None,
                                verbose=0)
    url = "https://www.dropbox.com/s/eujtyfb5zem1mim/EN-GOOGLE.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-GOOGLE.txt", url, {})],
                            resume=True,
                            verbose=0)[0]

    with open(raw_data, "r") as f:
        L = f.read().splitlines()

    questions = defaultdict(list)
    answers = defaultdict(list)
    category = None
    for l in L:
        if l.startswith(":"):
            category = l.split()[1]
        else:
            words = l.split()
            questions[category].append(words[0:3])
            answers[category].append(words[3])

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
     'city-in-state','family', 'gram9-plural-verbs', 'gram2-opposite',
     'currency', 'gram4-superlative', 'gram6-nationality-adjective', 'gram7-past-tense',
     'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])

    categories_high_level = {
        "syntactic": [c for c in questions if c.startswith("gram")],
        "semantic": [c for c in questions if not c.startswith("gram")]
    }

    return Bunch(X=_change_list_to_np(questions),
                 y=_change_list_to_np(answers),
                 categories_high_level=categories_high_level)