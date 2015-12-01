"""
 Functions for fetching data
"""

from sklearn.datasets.base import Bunch
from .utils import _get_dataset_dir, _fetch_files
from collections import defaultdict
import glob, os

def fetch_msr_analogy():
    """
    Fetch MSR dataset for testing performance on syntactic analogies

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'question': dictionary keyed on category with lists of space
        delimited question words
        'answers': dictionary keyed on category with the answer

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
        questions[words[3]].append(" ".join(words[0:3]))
        answers[words[3]] = words[4]

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['VBD_VBZ', 'VB_VBD', 'VBZ_VBD',
        'VBZ_VB', 'NNPOS_NN', 'JJR_JJS', 'JJS_JJR', 'NNS_NN', 'JJR_JJ',
        'NN_NNS', 'VB_VBZ', 'VBD_VB', 'JJS_JJ', 'NN_NNPOS', 'JJ_JJS', 'JJ_JJR'])

    return Bunch(questions=questions, answers=answers)

def fetch_semeval_2012_2(which="all"):
    """
    Parameters
    -------
    which : "all", "train" or "test"

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'questions': dictionary keyed on category. Each entry is list of question 'prototype' words
        (please refer to description) and list of question pairs
        'golden_scores': released during competition golden scores for words
        'platinium_scores': release after competition improved golden scores for words
        'categories_names': dictionary keyed on category. Each entry is a human readable name of
        category.
        'categories_descriptions': dictionary keyed on category. Each entry is a human readable description of
        category.

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """
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
    else:
        raise RuntimeError("Unrecognized which argument")

    questions = {}
    golden_scores = {}
    platinium_scores = {}
    categories_names = {}
    categories_descriptions = {}
    for f in files:
        meta = open(f[0:-4] + "_meta.txt").read().splitlines()[1].split(",")
        c = meta[0] + "_" + meta[1]
        categories_names[c] = meta[2] + "_" + meta[3]
        categories_descriptions[c] = meta[4]

        platinium = open(os.path.dirname(f) + "/pl-" + os.path.basename(f)).read().splitlines()
        golden = open(f).read().splitlines()

        assert platinium[0] == golden[0]

        questions[c] = [platinium[0].split(","), []]
        golden_scores[c] = {}
        platinium_scores[c] = {}

        for line_pl in platinium[1:]:
            word_pair, score = line_pl.split()
            questions[c][1].append(word_pair)
            platinium_scores[c][word_pair] = score

        for line_g in golden[1:]:
            word_pair, score = line_g.split()
            golden_scores[c][word_pair] = score

    return Bunch(questions=questions, golden_scores=golden_scores, platinium_scores=platinium_scores,
                categories_names=categories_names, categories_descriptions=categories_descriptions)

def fetch_wordrep(subsample=None, rng=None):
    """
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'word_pairs': dictionary keyed on category with lists of space
        delimited pairs of words. You can form questions by taking any pair of pairs of words
        'answers': dictionary keyed on category with the answer
        'categories_high_level': dictionary keyed on higher level category that
        provides coarse grained grouping of categories

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

    """
    data_dir = _get_dataset_dir("analogy", verbose=3)

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
        if not len(word_pairs[c]):
            print c

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

    return Bunch(categories_high_level=categories_high_level, word_pairs=word_pairs)

def fetch_google_analogy():
    """
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'question': dictionary keyed on category with lists of space
        delimited question words
        'answers': dictionary keyed on category with the answer
        'categories_high_level': dictionary keyed on higher level category that
        provides coarse grained grouping of categories

    References
    ----------
    TODO: Add Indian Pines references

    Notes
    -----
    TODO: Add notes

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
            questions[category].append(" ".join(words[0:3]))
            answers[category] = words[3]

    assert questions.keys() == answers.keys()
    assert set(questions.keys()) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
     'city-in-state','family', 'gram9-plural-verbs', 'gram2-opposite',
     'currency', 'gram4-superlative', 'gram6-nationality-adjective', 'gram7-past-tense',
     'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])

    categories_high_level = {
        "syntactic": [c for c in questions if c.startswith("gram")],
        "semantic": [c for c in questions if not c.startswith("gram")]
    }

    return Bunch(questions=questions, answers=answers, categories_high_level=categories_high_level)