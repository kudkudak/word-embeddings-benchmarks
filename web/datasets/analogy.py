# -*- coding: utf-8 -*-

"""
 Functions for fetching analogy datasets
"""

from collections import defaultdict
import glob
import os
import numpy as np

from sklearn.utils import check_random_state

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_files, _change_list_to_np
from ..utils import standardize_string


def fetch_wordrep(subsample=None, rng=None):
    """
    Fetch  MSR WordRep dataset for testing both syntactic and semantic dataset

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word pairs
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    This dataset is too big to calculate and store all word analogy quadruples, this is
    why it returns word paris

    """
    data_dir = _get_dataset_dir("analogy", verbose=0)
    path = _fetch_files(data_dir, [("EN-WORDREP",
                                    "https://www.dropbox.com/sh/5k78h9gllvc44vt/AAALLQq-Bge605OIMlmGBbNJa?dl=1",
                                    {'uncompress': True, "move": "EN-WORDREP/EN-WORDREP.zip"})],
                        verbose=0)[0]

    wikipedia_dict = glob.glob(os.path.join(path, "Pairs_from_Wikipedia_and_Dictionary/*.txt"))
    wordnet = glob.glob(os.path.join(path, "Pairs_from_WordNet/*.txt"))

    # This dataset is too big to calculate and store all word analogy quadruples
    word_pairs = []
    category = []
    category_high_level = []

    files = wikipedia_dict + wordnet

    for file_name in files:
        c = os.path.basename(file_name).split(".")[0].split("-")[1].lower()
        with open(file_name, "r") as f:
            for l in f.read().splitlines():
                word_pairs.append(standardize_string(l).split())
                category.append(c)
                category_high_level.append("wikipedia-dict" if file_name in wikipedia_dict else "wordnet")

    if subsample:
        assert 0 <= subsample <= 1.0
        rng = check_random_state(rng)
        ids = rng.choice(range(len(word_pairs)), int(subsample * len(word_pairs)), replace=False)
        word_pairs = [word_pairs[i] for i in ids]
        category = [category[i] for i in ids]
        category_high_level = [category_high_level[i] for i in ids]

    return Bunch(category_high_level=np.array(category_high_level),
                 X=np.array(word_pairs),
                 category=np.array(category))


def fetch_google_analogy():
    """
    Fetch Google dataset for testing both semantic and syntactic analogies.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (semantic/syntactic)

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    This dataset is a subset of WordRep dataset.

    """

    data_dir = _get_dataset_dir('analogy/EN-GOOGLE', data_dir=None, verbose=0)
    url = "https://www.dropbox.com/s/eujtyfb5zem1mim/EN-GOOGLE.txt?dl=1"
    raw_data = _fetch_files(data_dir, [("EN-GOOGLE.txt", url, {})], verbose=0)[0]

    with open(raw_data, "r") as f:
        L = f.read().splitlines()

    # Simple 4 word analogy questions with categories
    questions = []
    answers = []
    category = []
    cat = None
    for l in L:
        if l.startswith(":"):
            cat =l.lower().split()[1]
        else:
            words =  standardize_string(l).split()
            questions.append(words[0:3])
            answers.append(words[3])
            category.append(cat)

    assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
                                         'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
                                         'currency', 'gram4-superlative', 'gram6-nationality-adjective',
                                         'gram7-past-tense',
                                         'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])


    syntactic = set([c for c in set(category) if c.startswith("gram")])
    category_high_level = []
    for cat in category:
         category_high_level.append("syntactic" if cat in syntactic else "semantic")

    return Bunch(X=np.vstack(questions),
                 y=np.hstack(answers),
                 category=np.hstack(category),
                 category_high_level=np.hstack(category_high_level))


def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of word questions
        'y': vector of answers
        'category': name of category
        'category_high_level': name of high level category (noun/adjective/verb)

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """

    data_dir = _get_dataset_dir('analogy/EN-MSR', data_dir=None, verbose=0)
    url = "https://www.dropbox.com/s/ne0fib302jqbatw/EN-MSR.txt?dl=1"
    path = _fetch_files(data_dir, [("EN-MSR.txt", url, {})], verbose=0)[0]

    with open(path, "r") as f:
        L = f.read().splitlines()

    # Typical 4 words analogy questions
    questions = []
    answers = []
    category = []
    for l in L:
        words = standardize_string(l).split()
        questions.append(words[0:3])
        answers.append(words[4])
        category.append(words[3])

    verb = set([c for c in set(category) if c.startswith("VB")])
    noun = set([c for c in set(category) if c.startswith("NN")])
    category_high_level = []
    for cat in category:
         if cat in verb:
             category_high_level.append("verb")
         elif cat in noun:
             category_high_level.append("noun")
         else:
             category_high_level.append("adjective")

    assert set([c.upper() for c in category]) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
                                         'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
                                         'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(X=np.vstack(questions), y=np.hstack(answers), category=np.hstack(category),
                 category_high_level=np.hstack(category_high_level))


# TODO: rewrite to a more standarized version
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
    path = _fetch_files(data_dir, [("EN-SEMVAL-2012-2",
                                    "https://www.dropbox.com/sh/yjzunhyqzsu1z47/AAAjyWDfP_ZAkmmNus4YBAEHa?dl=1",
                                     {
                                        'uncompress': True,
                                        "move": "EN-SEMVAL-2012-2/EN-SEMVAL-2012-2.zip"
                                     })],
                        verbose=0)[0]

    train_files = set(glob.glob(os.path.join(path, "train*.txt"))) - \
                  set(glob.glob(os.path.join(path, "train*_meta.txt")))
    test_files = set(glob.glob(os.path.join(path, "test*.txt"))) - \
                 set(glob.glob(os.path.join(path, "test*_meta.txt")))

    if which == "train":
        files = train_files
    elif which == "test":
        files = test_files
    elif which == "all":
        files = train_files.union(test_files)

    # Every question is formed as similarity to analogy category that is
    # posed as a list of 3 prototype word pairs
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
            word_pair, score = standardize_string(line_pl).split()
            questions[c].append(word_pair.split(":"))
            platinium_scores[c][word_pair] = score

        for line_g in golden[1:]:
            word_pair, score = standardize_string(line_g).split()
            golden_scores[c][word_pair] = score

    return Bunch(X_prot=_change_list_to_np(questions),
                 X=_change_list_to_np(questions),
                 y=scores[which_scoring],
                 categories_names=categories_names,
                 categories_descriptions=categories_descriptions)


